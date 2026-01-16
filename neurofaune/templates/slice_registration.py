"""
Slice-wise registration for thick-slice templates to isotropic atlas.

This module handles registration of templates acquired with thick coronal slices
(e.g., 8mm) to isotropic atlases (e.g., SIGMA at 1mm). Standard 3D non-linear
registration fails in this case due to the extreme anisotropy mismatch.

The approach:
1. Reorient template to match atlas orientation (transpose + flip as needed)
2. For each coronal slice in the template, extract the corresponding coronal
   region from the atlas (averaging across the slice thickness)
3. Perform 2D affine registration for each slice pair
4. Store per-slice transforms for atlas label propagation
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def reorient_template_to_sigma(
    template_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_voxel_size: float = 1.5,
    save_1mm: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorient a thick-slice template to match SIGMA atlas orientation.

    The BPA-Rat templates are acquired with thick coronal slices (8mm) where:
    - Original X axis = Left-Right
    - Original Y axis = Inferior-Superior
    - Original Z axis = Anterior-Posterior (thick coronal slices)

    SIGMA atlas has:
    - X axis = Left-Right
    - Y axis = Anterior-Posterior
    - Z axis = Inferior-Superior

    This function performs:
    1. transpose(0, 2, 1) to swap Y and Z axes
    2. flip(axis=2) to correct the I-S direction
    3. Resample to target voxel size (default 1.5mm to match SIGMA)

    Parameters
    ----------
    template_path : Path
        Path to original template NIfTI file
    output_path : Path, optional
        Path to save reoriented template. If None, returns data without saving.
    target_voxel_size : float
        Target isotropic voxel size in mm (default 1.5 to match scaled SIGMA)
    save_1mm : bool
        If True and output_path is provided, also save a 1mm version

    Returns
    -------
    tuple
        (reoriented_data, affine) - the reoriented array and its affine matrix
    """
    template_path = Path(template_path)
    template_img = nib.load(template_path)
    template_data = template_img.get_fdata()
    template_voxels = template_img.header.get_zooms()

    # Step 1: Reorient by transposing Y and Z
    # This maps: X->X, Y->Z, Z->Y
    template_reoriented = np.transpose(template_data, (0, 2, 1))

    # Step 2: Flip the new Z axis (originally Y) to correct I-S direction
    template_reoriented = np.flip(template_reoriented, axis=2)

    # Voxel sizes after transpose: (X, Z_orig, Y_orig) = (1.25, 8.0, 1.25)
    reoriented_voxels = (template_voxels[0], template_voxels[2], template_voxels[1])

    # Step 3: Resample to target voxel size
    zoom_factors = [reoriented_voxels[i] / target_voxel_size for i in range(3)]
    template_resampled = zoom(template_reoriented, zoom_factors, order=1)

    # Create affine for RAS orientation at target voxel size
    affine = np.eye(4) * target_voxel_size
    affine[3, 3] = 1.0

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        new_img = nib.Nifti1Image(template_resampled.astype(np.float32), affine)
        nib.save(new_img, output_path)

        if save_1mm:
            # Also save 1mm version
            zoom_to_1mm = [reoriented_voxels[i] / 1.0 for i in range(3)]
            template_1mm = zoom(template_reoriented, zoom_to_1mm, order=1)
            affine_1mm = np.eye(4)

            output_1mm = output_path.parent / output_path.name.replace('.nii', '_1mm.nii')
            new_img_1mm = nib.Nifti1Image(template_1mm.astype(np.float32), affine_1mm)
            nib.save(new_img_1mm, output_1mm)

    return template_resampled, affine


def reorient_sigma_to_study(
    sigma_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    is_labels: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorient SIGMA atlas to match BPA-Rat study native orientation.

    This is the inverse of reorient_template_to_sigma(). Instead of reorienting
    each template/image, we reorient the atlas once to match the study's native
    acquisition space.

    SIGMA atlas orientation: X=L-R, Y=A-P, Z=I-S
    Study native orientation: X=L-R, Y=I-S, Z=A-P

    Transformation:
    1. transpose(0, 2, 1) to swap Y and Z axes
    2. flip(axis=0) to correct L-R direction
    3. flip(axis=1) to correct the new Y (I-S) direction

    Parameters
    ----------
    sigma_path : Path
        Path to SIGMA atlas NIfTI file
    output_path : Path, optional
        Path to save reoriented atlas. If None, returns data without saving.
    is_labels : bool
        If True, treat as label image (use int16, nearest neighbor)

    Returns
    -------
    tuple
        (reoriented_data, affine) - the reoriented array and its affine matrix
    """
    sigma_path = Path(sigma_path)
    sigma_img = nib.load(sigma_path)
    sigma_data = sigma_img.get_fdata()
    sigma_voxels = sigma_img.header.get_zooms()

    # Step 1: Swap Y and Z axes
    reoriented = np.transpose(sigma_data, (0, 2, 1))

    # Step 2: Flip X axis
    reoriented = np.flip(reoriented, axis=0)

    # Step 3: Flip Y axis (the new Y, which was Z)
    reoriented = np.flip(reoriented, axis=1)

    # Voxel sizes after transpose: (X, Z_orig, Y_orig)
    new_voxel_sizes = (sigma_voxels[0], sigma_voxels[2], sigma_voxels[1])
    affine = np.diag([new_voxel_sizes[0], new_voxel_sizes[1], new_voxel_sizes[2], 1.0])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if is_labels:
            new_img = nib.Nifti1Image(reoriented.astype(np.int16), affine)
        else:
            new_img = nib.Nifti1Image(reoriented.astype(np.float32), affine)
        nib.save(new_img, output_path)

    return reoriented, affine


def setup_study_atlas(
    sigma_base_path: Union[str, Path],
    study_atlas_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Set up study-space SIGMA atlas by reorienting all files to match study orientation.

    This function:
    1. Reorients all SIGMA InVivo atlas files to match the study's native acquisition
    2. Saves them to the study atlas directory
    3. Optionally updates the config file with the new paths

    Parameters
    ----------
    sigma_base_path : Path
        Path to SIGMA_scaled directory (e.g., /mnt/arborea/atlases/SIGMA_scaled)
    study_atlas_dir : Path
        Output directory for study-space atlas (e.g., {study_root}/atlas/SIGMA_study_space)
    config_path : Path, optional
        Path to study config YAML file to update. If provided, adds study_space_atlas
        section with file paths.

    Returns
    -------
    dict
        Dictionary mapping file types to output paths

    Example
    -------
    >>> paths = setup_study_atlas(
    ...     sigma_base_path="/mnt/arborea/atlases/SIGMA_scaled",
    ...     study_atlas_dir="/mnt/arborea/bpa-rat/atlas/SIGMA_study_space",
    ...     config_path="/mnt/arborea/bpa-rat/config.yaml"
    ... )
    """
    import json
    import yaml

    sigma_base_path = Path(sigma_base_path)
    study_atlas_dir = Path(study_atlas_dir)
    study_atlas_dir.mkdir(parents=True, exist_ok=True)

    # Define files to reorient
    # Format: (source_subpath, output_name, is_labels)
    files_to_process = [
        # Templates and masks
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Template.nii",
         "SIGMA_InVivo_Brain_Template.nii.gz", False),
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Template_Masked.nii",
         "SIGMA_InVivo_Brain_Template_Masked.nii.gz", False),
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Mask.nii",
         "SIGMA_InVivo_Brain_Mask.nii.gz", False),
        # Tissue probability maps
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_GM.nii",
         "SIGMA_InVivo_GM.nii.gz", False),
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_WM.nii",
         "SIGMA_InVivo_WM.nii.gz", False),
        ("SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_CSF.nii",
         "SIGMA_InVivo_CSF.nii.gz", False),
        # Anatomical atlas (label map)
        ("SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/InVivo_Atlas/SIGMA_InVivo_Anatomical_Brain_Atlas.nii",
         "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz", True),
    ]

    output_paths = {}
    metadata = {
        "description": "SIGMA atlas reoriented to study native space",
        "transformation": {
            "step1": "transpose(0, 2, 1) - swap Y and Z axes",
            "step2": "flip(axis=0) - flip X axis",
            "step3": "flip(axis=1) - flip Y axis"
        },
        "files": {}
    }

    print(f"Setting up study-space SIGMA atlas in: {study_atlas_dir}")

    for source_subpath, output_name, is_labels in files_to_process:
        source_path = sigma_base_path / source_subpath
        output_path = study_atlas_dir / output_name

        if not source_path.exists():
            print(f"  Warning: Source file not found: {source_path}")
            continue

        print(f"  Reorienting: {output_name} (labels={is_labels})")

        # Load source to get shape info
        source_img = nib.load(source_path)
        source_shape = source_img.shape

        # Reorient and save
        reoriented_data, affine = reorient_sigma_to_study(
            source_path, output_path, is_labels=is_labels
        )

        # Track output
        file_key = output_name.replace('.nii.gz', '').replace('.nii', '')
        output_paths[file_key] = output_path

        # Record metadata
        metadata["files"][source_path.name] = {
            "source": str(source_path),
            "output": str(output_path),
            "original_shape": list(source_shape),
            "reoriented_shape": list(reoriented_data.shape),
            "is_labels": is_labels
        }

        if is_labels:
            n_labels = len(np.unique(reoriented_data.astype(int)))
            metadata["files"][source_path.name]["unique_labels"] = n_labels

    # Save metadata
    metadata_path = study_atlas_dir / "atlas_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    # Update config file if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            _update_config_with_atlas(config_path, study_atlas_dir, output_paths)
        else:
            print(f"  Warning: Config file not found: {config_path}")
            print(f"  Creating new config section to add manually...")
            _print_config_section(study_atlas_dir, output_paths)

    print(f"✓ Study-space SIGMA atlas setup complete")
    return output_paths


def _update_config_with_atlas(
    config_path: Path,
    study_atlas_dir: Path,
    output_paths: Dict[str, Path]
) -> None:
    """Update the config YAML file with study-space atlas paths."""
    import yaml

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Add or update study_space_atlas section under atlas
    if 'atlas' not in config:
        config['atlas'] = {}

    config['atlas']['study_space'] = {
        'base_path': str(study_atlas_dir),
        'template': str(output_paths.get('SIGMA_InVivo_Brain_Template', '')),
        'template_masked': str(output_paths.get('SIGMA_InVivo_Brain_Template_Masked', '')),
        'brain_mask': str(output_paths.get('SIGMA_InVivo_Brain_Mask', '')),
        'gm_prob': str(output_paths.get('SIGMA_InVivo_GM', '')),
        'wm_prob': str(output_paths.get('SIGMA_InVivo_WM', '')),
        'csf_prob': str(output_paths.get('SIGMA_InVivo_CSF', '')),
        'parcellation': str(output_paths.get('SIGMA_InVivo_Anatomical_Brain_Atlas', '')),
    }

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Updated config: {config_path}")


def _print_config_section(study_atlas_dir: Path, output_paths: Dict[str, Path]) -> None:
    """Print the config section to add manually."""
    print("\n# Add this section to your config.yaml under 'atlas:'")
    print("  study_space:")
    print(f"    base_path: \"{study_atlas_dir}\"")
    print(f"    template: \"{output_paths.get('SIGMA_InVivo_Brain_Template', '')}\"")
    print(f"    template_masked: \"{output_paths.get('SIGMA_InVivo_Brain_Template_Masked', '')}\"")
    print(f"    brain_mask: \"{output_paths.get('SIGMA_InVivo_Brain_Mask', '')}\"")
    print(f"    gm_prob: \"{output_paths.get('SIGMA_InVivo_GM', '')}\"")
    print(f"    wm_prob: \"{output_paths.get('SIGMA_InVivo_WM', '')}\"")
    print(f"    csf_prob: \"{output_paths.get('SIGMA_InVivo_CSF', '')}\"")
    print(f"    parcellation: \"{output_paths.get('SIGMA_InVivo_Anatomical_Brain_Atlas', '')}\"")


def get_slice_geometry(img: nib.Nifti1Image) -> Dict:
    """
    Analyze image geometry to determine slice orientation and spacing.

    Parameters
    ----------
    img : Nifti1Image
        Input image

    Returns
    -------
    dict
        Dictionary with geometry info:
        - 'shape': tuple of dimensions
        - 'voxel_size': tuple of voxel sizes in mm
        - 'thick_axis': axis with largest voxel size (0, 1, or 2)
        - 'thick_voxel': voxel size of thick axis in mm
        - 'n_slices': number of slices along thick axis
        - 'orientation': tuple of axis codes (e.g., ('R', 'A', 'S'))
    """
    shape = img.shape[:3]
    voxel_size = img.header.get_zooms()[:3]
    orientation = nib.aff2axcodes(img.affine)

    # Find the thick-slice axis
    thick_axis = int(np.argmax(voxel_size))

    return {
        'shape': shape,
        'voxel_size': voxel_size,
        'thick_axis': thick_axis,
        'thick_voxel': float(voxel_size[thick_axis]),
        'n_slices': shape[thick_axis],
        'orientation': orientation
    }


def extract_coronal_slice_template(
    data: np.ndarray,
    slice_idx: int,
    thick_axis: int = 2
) -> np.ndarray:
    """
    Extract a coronal slice from the template.

    For thick-slice coronal acquisitions, the slice is in the XY plane
    (assuming thick_axis=2, which is Z).

    Parameters
    ----------
    data : np.ndarray
        3D image data
    slice_idx : int
        Slice index along thick axis
    thick_axis : int
        Axis of thick slices (default 2 for Z)

    Returns
    -------
    np.ndarray
        2D slice
    """
    if thick_axis == 0:
        return data[slice_idx, :, :]
    elif thick_axis == 1:
        return data[:, slice_idx, :]
    else:  # thick_axis == 2
        return data[:, :, slice_idx]


def extract_coronal_slab_atlas(
    data: np.ndarray,
    center_mm: float,
    thickness_mm: float,
    voxel_size: float,
    coronal_axis: int = 1
) -> np.ndarray:
    """
    Extract and average a coronal slab from the atlas.

    For isotropic atlases, coronal slices are typically in the XZ plane
    (Y is the anterior-posterior axis).

    Parameters
    ----------
    data : np.ndarray
        3D atlas data
    center_mm : float
        Center of slab in mm along coronal axis
    thickness_mm : float
        Thickness of slab in mm to average
    voxel_size : float
        Voxel size of atlas in mm
    coronal_axis : int
        Axis corresponding to anterior-posterior (default 1 for Y)

    Returns
    -------
    np.ndarray
        2D averaged coronal slice
    """
    # Convert mm to voxel indices
    center_vox = int(center_mm / voxel_size)
    half_thickness_vox = int((thickness_mm / 2) / voxel_size)

    # Ensure within bounds
    start_vox = max(0, center_vox - half_thickness_vox)
    end_vox = min(data.shape[coronal_axis], center_vox + half_thickness_vox + 1)

    if start_vox >= end_vox:
        # Return zeros if completely out of bounds
        if coronal_axis == 0:
            return np.zeros((data.shape[1], data.shape[2]))
        elif coronal_axis == 1:
            return np.zeros((data.shape[0], data.shape[2]))
        else:
            return np.zeros((data.shape[0], data.shape[1]))

    # Extract and average
    if coronal_axis == 0:
        slab = data[start_vox:end_vox, :, :]
        return np.mean(slab, axis=0)
    elif coronal_axis == 1:
        slab = data[:, start_vox:end_vox, :]
        return np.mean(slab, axis=1)
    else:
        slab = data[:, :, start_vox:end_vox]
        return np.mean(slab, axis=2)


def find_brain_extent(data: np.ndarray, axis: int, threshold: float = 0.1) -> Tuple[int, int]:
    """
    Find the extent of brain along a given axis.

    Parameters
    ----------
    data : np.ndarray
        3D image data
    axis : int
        Axis to find extent along
    threshold : float
        Threshold as fraction of max intensity

    Returns
    -------
    tuple
        (start_idx, end_idx) of brain extent
    """
    mask = data > threshold * data.max()

    # Sum along all other axes
    axes_to_sum = tuple(i for i in range(3) if i != axis)
    projection = mask.sum(axis=axes_to_sum)

    nonzero = np.where(projection > 0)[0]
    if len(nonzero) == 0:
        return 0, data.shape[axis] - 1

    return int(nonzero[0]), int(nonzero[-1])


def compute_slice_correspondence(
    template_geom: Dict,
    atlas_geom: Dict,
    template_data: np.ndarray,
    atlas_data: np.ndarray,
    template_coronal_axis: int = 2,
    atlas_coronal_axis: int = 1
) -> List[Dict]:
    """
    Compute which atlas region corresponds to each template slice.

    Uses brain extent matching to properly map between template and atlas
    even when FOV sizes are different.

    Parameters
    ----------
    template_geom : dict
        Template geometry from get_slice_geometry()
    atlas_geom : dict
        Atlas geometry from get_slice_geometry()
    template_data : np.ndarray
        Template image data (for brain extent detection)
    atlas_data : np.ndarray
        Atlas image data (for brain extent detection)
    template_coronal_axis : int
        Template axis for coronal slices (default 2 for Z)
    atlas_coronal_axis : int
        Atlas axis for coronal slices (default 1 for Y)

    Returns
    -------
    list of dict
        For each template slice: {'template_idx', 'atlas_center_mm', 'atlas_thickness_mm'}
    """
    # Get voxel sizes
    template_voxel = template_geom['voxel_size'][template_coronal_axis]
    atlas_voxel = atlas_geom['voxel_size'][atlas_coronal_axis]

    # Find brain extent in each image
    template_start, template_end = find_brain_extent(template_data, template_coronal_axis)
    atlas_start, atlas_end = find_brain_extent(atlas_data, atlas_coronal_axis)

    # Convert to mm
    template_start_mm = template_start * template_voxel
    template_end_mm = template_end * template_voxel
    template_extent_mm = template_end_mm - template_start_mm

    atlas_start_mm = atlas_start * atlas_voxel
    atlas_end_mm = atlas_end * atlas_voxel
    atlas_extent_mm = atlas_end_mm - atlas_start_mm

    # Calculate scale factor (atlas extent / template extent)
    scale_factor = atlas_extent_mm / template_extent_mm if template_extent_mm > 0 else 1.0

    n_slices = template_geom['shape'][template_coronal_axis]

    correspondences = []
    for i in range(n_slices):
        # Template slice center in mm
        template_slice_mm = (i + 0.5) * template_voxel

        # Calculate relative position within brain extent (0 to 1)
        if template_extent_mm > 0:
            relative_pos = (template_slice_mm - template_start_mm) / template_extent_mm
        else:
            relative_pos = 0.5

        # Map to atlas position
        atlas_mm = atlas_start_mm + relative_pos * atlas_extent_mm

        # Scale thickness proportionally
        atlas_thickness = template_voxel * scale_factor

        correspondences.append({
            'template_idx': i,
            'template_mm': template_slice_mm,
            'atlas_center_mm': atlas_mm,
            'atlas_thickness_mm': atlas_thickness,
            'relative_position': relative_pos,
            'in_brain': 0 <= relative_pos <= 1
        })

    return correspondences


def register_2d_slices(
    template_slice: np.ndarray,
    atlas_slice: np.ndarray,
    output_prefix: Path,
    use_ants: bool = True
) -> Dict:
    """
    Register two 2D slices using affine transformation.

    Parameters
    ----------
    template_slice : np.ndarray
        2D template slice (moving)
    atlas_slice : np.ndarray
        2D atlas slice (fixed)
    output_prefix : Path
        Prefix for output files
    use_ants : bool
        Use ANTs for registration (if False, use scipy/opencv fallback)

    Returns
    -------
    dict
        Registration results with transform matrix and quality metrics
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if use_ants:
        return _register_2d_ants(template_slice, atlas_slice, output_prefix)
    else:
        return _register_2d_scipy(template_slice, atlas_slice)


def _register_2d_ants(
    template_slice: np.ndarray,
    atlas_slice: np.ndarray,
    output_prefix: Path
) -> Dict:
    """2D registration using ANTs (using 3D with single slice)."""
    import tempfile

    # Resample template slice to match atlas slice dimensions for registration
    if template_slice.shape != atlas_slice.shape:
        zoom_factors = [atlas_slice.shape[i] / template_slice.shape[i] for i in range(2)]
        template_resampled = zoom(template_slice, zoom_factors, order=1)
    else:
        template_resampled = template_slice

    # Save as 3D NIfTI files with single slice (ANTs 2D mode has issues with nibabel)
    # Use 3D mode instead with a thin volume
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 3D arrays with single slice
        template_3d = template_resampled[:, :, np.newaxis]
        atlas_3d = atlas_slice[:, :, np.newaxis]

        # Use identity affine with 1mm voxels
        affine = np.eye(4)
        template_nii = nib.Nifti1Image(template_3d.astype(np.float32), affine)
        atlas_nii = nib.Nifti1Image(atlas_3d.astype(np.float32), affine)

        template_path = tmpdir / 'template_slice.nii.gz'
        atlas_path = tmpdir / 'atlas_slice.nii.gz'
        nib.save(template_nii, template_path)
        nib.save(atlas_nii, atlas_path)

        # Run ANTs 3D registration on 2D slices (rigid + affine, no SyN)
        # This works because the Z dimension is 1 voxel
        cmd = [
            'antsRegistration',
            '-d', '3',  # Use 3D mode with single-slice images
            '-v', '0',
            '-o', str(output_prefix),
            '-t', 'Rigid[0.1]',
            '-m', f'MI[{atlas_path},{template_path},1,32,Regular,0.25]',
            '-c', '[500x250x100,1e-6,10]',
            '-f', '4x2x1',
            '-s', '2x1x0',
            '-t', 'Affine[0.1]',
            '-m', f'MI[{atlas_path},{template_path},1,32,Regular,0.25]',
            '-c', '[500x250x100,1e-6,10]',
            '-f', '4x2x1',
            '-s', '2x1x0',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {'success': False, 'error': result.stderr}

        # Read the transform
        transform_path = Path(str(output_prefix) + '0GenericAffine.mat')

        # Apply transform to get warped image for quality check
        warped_path = tmpdir / 'warped.nii.gz'
        apply_cmd = [
            'antsApplyTransforms',
            '-d', '3',  # Match registration dimension
            '-i', str(template_path),
            '-r', str(atlas_path),
            '-o', str(warped_path),
            '-t', str(transform_path),
            '-n', 'Linear'
        ]
        subprocess.run(apply_cmd, capture_output=True)

        # Calculate correlation as quality metric
        if warped_path.exists():
            warped_data = nib.load(warped_path).get_fdata()[:, :, 0]
            atlas_ref = nib.load(atlas_path).get_fdata()[:, :, 0]
            correlation = np.corrcoef(
                atlas_ref.flatten(),
                warped_data.flatten()
            )[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        return {
            'success': True,
            'transform_path': str(transform_path),
            'correlation': correlation
        }


def _register_2d_scipy(
    template_slice: np.ndarray,
    atlas_slice: np.ndarray
) -> Dict:
    """Fallback 2D registration using scipy/skimage."""
    from scipy.ndimage import affine_transform
    from scipy.optimize import minimize

    # Resample template to atlas size
    if template_slice.shape != atlas_slice.shape:
        zoom_factors = [atlas_slice.shape[i] / template_slice.shape[i] for i in range(2)]
        template_resampled = zoom(template_slice, zoom_factors, order=1)
    else:
        template_resampled = template_slice

    # Normalize
    template_norm = (template_resampled - template_resampled.mean()) / (template_resampled.std() + 1e-10)
    atlas_norm = (atlas_slice - atlas_slice.mean()) / (atlas_slice.std() + 1e-10)

    def cost_function(params):
        """Negative correlation after affine transform."""
        # params: [angle, tx, ty, scale_x, scale_y]
        angle, tx, ty, sx, sy = params

        # Build affine matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        scale = np.array([[sx, 0], [0, sy]])
        matrix = rotation @ scale
        offset = [tx, ty]

        # Apply transform
        transformed = affine_transform(template_norm, matrix, offset=offset, order=1)

        # Negative correlation (we minimize)
        corr = np.corrcoef(atlas_norm.flatten(), transformed.flatten())[0, 1]
        return -corr if not np.isnan(corr) else 0

    # Initial parameters: no rotation, no translation, unit scale
    x0 = [0, 0, 0, 1, 1]
    bounds = [(-np.pi/4, np.pi/4), (-50, 50), (-50, 50), (0.8, 1.2), (0.8, 1.2)]

    result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds)

    angle, tx, ty, sx, sy = result.x
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return {
        'success': True,
        'transform_matrix': np.array([
            [sx * cos_a, -sy * sin_a, tx],
            [sx * sin_a, sy * cos_a, ty],
            [0, 0, 1]
        ]),
        'correlation': -result.fun
    }


def slice_wise_registration(
    template_path: Path,
    atlas_path: Path,
    output_dir: Path,
    template_coronal_axis: int = 2,
    atlas_coronal_axis: int = 1,
    use_ants: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Perform slice-wise 2D registration between template and atlas.

    Parameters
    ----------
    template_path : Path
        Path to thick-slice template
    atlas_path : Path
        Path to isotropic atlas
    output_dir : Path
        Directory for output transforms and results
    template_coronal_axis : int
        Template axis for coronal slices (default 2)
    atlas_coronal_axis : int
        Atlas axis for coronal slices (default 1)
    use_ants : bool
        Use ANTs for 2D registration
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results including per-slice transforms and quality metrics
    """
    template_path = Path(template_path)
    atlas_path = Path(atlas_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    template_img = nib.load(template_path)
    atlas_img = nib.load(atlas_path)

    template_data = np.squeeze(template_img.get_fdata())
    atlas_data = np.squeeze(atlas_img.get_fdata())

    # Get geometry
    template_geom = get_slice_geometry(template_img)
    atlas_geom = get_slice_geometry(atlas_img)

    if verbose:
        print(f"Template: {template_geom['shape']}, thick axis={template_geom['thick_axis']}")
        print(f"Atlas: {atlas_geom['shape']}")

    # Compute slice correspondences
    correspondences = compute_slice_correspondence(
        template_geom, atlas_geom,
        template_data, atlas_data,
        template_coronal_axis, atlas_coronal_axis
    )

    # Register each slice
    results = {
        'template_path': str(template_path),
        'atlas_path': str(atlas_path),
        'n_slices': len(correspondences),
        'slices': []
    }

    atlas_voxel = atlas_geom['voxel_size'][atlas_coronal_axis]

    for corr in correspondences:
        slice_idx = corr['template_idx']

        if verbose:
            print(f"  Slice {slice_idx}/{len(correspondences)-1}...", end=' ')

        # Extract template slice
        template_slice = extract_coronal_slice_template(
            template_data, slice_idx, template_coronal_axis
        )

        # Extract corresponding atlas slab
        atlas_slice = extract_coronal_slab_atlas(
            atlas_data,
            corr['atlas_center_mm'],
            corr['atlas_thickness_mm'],
            atlas_voxel,
            atlas_coronal_axis
        )

        # Skip if atlas slice is empty (out of bounds)
        if atlas_slice.max() < 1e-6:
            if verbose:
                print("skipped (out of atlas bounds)")
            results['slices'].append({
                'slice_idx': slice_idx,
                'success': False,
                'reason': 'out_of_bounds'
            })
            continue

        # Register
        output_prefix = output_dir / f'slice_{slice_idx:03d}_'
        reg_result = register_2d_slices(
            template_slice, atlas_slice, output_prefix, use_ants
        )

        reg_result['slice_idx'] = slice_idx
        reg_result['correspondence'] = corr
        results['slices'].append(reg_result)

        if verbose:
            if reg_result['success']:
                print(f"r={reg_result.get('correlation', 0):.3f}")
            else:
                print(f"failed: {reg_result.get('error', 'unknown')[:50]}")

    # Summary statistics
    successful = [s for s in results['slices'] if s.get('success', False)]
    if successful:
        correlations = [s['correlation'] for s in successful if 'correlation' in s]
        results['mean_correlation'] = np.mean(correlations) if correlations else 0
        results['n_successful'] = len(successful)
    else:
        results['mean_correlation'] = 0
        results['n_successful'] = 0

    if verbose:
        print(f"\nCompleted: {results['n_successful']}/{results['n_slices']} slices")
        print(f"Mean correlation: {results['mean_correlation']:.3f}")

    return results


def propagate_labels_slice_wise(
    atlas_labels_path: Path,
    template_path: Path,
    registration_results: Dict,
    output_path: Path,
    atlas_coronal_axis: int = 1,
    template_coronal_axis: int = 2
) -> Path:
    """
    Propagate atlas labels to template space using slice-wise transforms.

    Parameters
    ----------
    atlas_labels_path : Path
        Path to atlas label image
    template_path : Path
        Path to template (for output geometry)
    registration_results : Dict
        Results from slice_wise_registration()
    output_path : Path
        Output path for propagated labels
    atlas_coronal_axis : int
        Atlas coronal axis
    template_coronal_axis : int
        Template coronal axis

    Returns
    -------
    Path
        Path to output label image
    """
    import tempfile

    atlas_labels_path = Path(atlas_labels_path)
    template_path = Path(template_path)
    output_path = Path(output_path)

    # Load images
    atlas_img = nib.load(atlas_labels_path)
    template_img = nib.load(template_path)

    atlas_data = np.squeeze(atlas_img.get_fdata())
    template_shape = template_img.shape[:3]

    # Initialize output label volume
    output_labels = np.zeros(template_shape, dtype=np.int32)

    atlas_geom = get_slice_geometry(atlas_img)
    atlas_voxel = atlas_geom['voxel_size'][atlas_coronal_axis]

    for slice_result in registration_results['slices']:
        if not slice_result.get('success', False):
            continue

        slice_idx = slice_result['slice_idx']
        corr = slice_result['correspondence']
        transform_path = slice_result.get('transform_path')

        if not transform_path or not Path(transform_path).exists():
            continue

        # Extract atlas label slab
        atlas_slab = extract_coronal_slab_atlas(
            atlas_data,
            corr['atlas_center_mm'],
            corr['atlas_thickness_mm'],
            atlas_voxel,
            atlas_coronal_axis
        )

        # For labels, use mode (most common value) instead of mean
        # Re-extract as integer labels
        center_vox = int(corr['atlas_center_mm'] / atlas_voxel)
        half_thick = int((corr['atlas_thickness_mm'] / 2) / atlas_voxel)
        start = max(0, center_vox - half_thick)
        end = min(atlas_data.shape[atlas_coronal_axis], center_vox + half_thick + 1)

        if start >= end:
            continue

        # Get the slab of labels
        if atlas_coronal_axis == 1:
            label_slab = atlas_data[:, start:end, :]
            # Take mode across the slab
            from scipy.stats import mode
            label_slice = mode(label_slab, axis=1, keepdims=False)[0]
        else:
            continue  # Not implemented for other axes

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Get template slice shape for reference
            template_data = np.squeeze(template_img.get_fdata())
            template_slice = extract_coronal_slice_template(
                template_data, slice_idx, template_coronal_axis
            )

            # Save label slice as 3D NIfTI
            label_3d = label_slice[:, :, np.newaxis].astype(np.float32)
            ref_3d = template_slice[:, :, np.newaxis].astype(np.float32)

            label_nii = nib.Nifti1Image(label_3d, np.eye(4))
            ref_nii = nib.Nifti1Image(ref_3d, np.eye(4))

            label_path = tmpdir / 'labels.nii.gz'
            ref_path = tmpdir / 'ref.nii.gz'
            warped_path = tmpdir / 'warped_labels.nii.gz'

            nib.save(label_nii, label_path)
            nib.save(ref_nii, ref_path)

            # Apply inverse transform (atlas -> template)
            cmd = [
                'antsApplyTransforms',
                '-d', '2',
                '-i', str(label_path),
                '-r', str(ref_path),
                '-o', str(warped_path),
                '-t', f'[{transform_path},1]',  # Inverse transform
                '-n', 'NearestNeighbor'  # For labels
            ]

            result = subprocess.run(cmd, capture_output=True)

            if warped_path.exists():
                warped_labels = nib.load(warped_path).get_fdata()[:, :, 0]

                # Insert into output volume
                if template_coronal_axis == 2:
                    output_labels[:, :, slice_idx] = warped_labels.astype(np.int32)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_nii = nib.Nifti1Image(output_labels, template_img.affine, template_img.header)
    nib.save(output_nii, output_path)

    return output_path
