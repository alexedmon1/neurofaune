#!/usr/bin/env python3
"""
Unit tests for atlas management module.

Tests AtlasManager functionality including:
- Initialization and configuration
- Template loading
- Slice extraction
- Tissue mask access
- ROI operations
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

from neurofaune.atlas import AtlasManager, extract_slices, get_slice_range
from neurofaune.atlas.slice_extraction import (
    extract_modality_slices,
    match_slice_geometry,
    get_slice_metadata
)


# Test fixtures
@pytest.fixture
def test_config():
    """Minimal configuration for testing."""
    return {
        'atlas': {
            'name': 'SIGMA',
            'base_path': '/mnt/arborea/atlases/SIGMA',
            'slice_definitions': {
                'dwi': {
                    'start': 15,
                    'end': 25,
                    'description': 'Hippocampus-centered for DTI'
                },
                'func': {
                    'start': 10,
                    'end': 35,
                    'description': 'Cortical coverage'
                },
                'anat': {
                    'start': 0,
                    'end': -1,
                    'description': 'Full brain'
                }
            }
        }
    }


@pytest.fixture
def mock_nifti_image():
    """Create a mock 3D NIfTI image for testing."""
    # Create 40x40x40 dummy image
    data = np.random.rand(40, 40, 40)
    affine = np.eye(4)
    affine[0, 0] = 0.15  # 150 micron voxels (typical for rodent)
    affine[1, 1] = 0.15
    affine[2, 2] = 0.15
    img = nib.Nifti1Image(data, affine)
    return img


class TestAtlasManager:
    """Tests for AtlasManager class."""

    def test_initialization(self, test_config):
        """Test AtlasManager initialization."""
        atlas = AtlasManager(test_config)
        assert atlas.atlas_type == 'invivo'
        assert atlas.base_path == Path('/mnt/arborea/atlases/SIGMA')
        assert len(atlas.roi_definitions) > 0  # Should load CSV

    def test_invalid_atlas_type(self, test_config):
        """Test that invalid atlas type raises error."""
        with pytest.raises(ValueError, match="atlas_type must be"):
            AtlasManager(test_config, atlas_type='invalid')

    def test_missing_atlas_path(self):
        """Test that missing atlas path raises error."""
        config = {'atlas': {'name': 'SIGMA'}}  # Missing base_path
        with pytest.raises(ValueError, match="base_path"):
            AtlasManager(config)

    def test_nonexistent_atlas_directory(self):
        """Test that nonexistent directory raises error."""
        config = {'atlas': {'name': 'SIGMA', 'base_path': '/nonexistent/path'}}
        with pytest.raises(FileNotFoundError):
            AtlasManager(config)

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_template(self, test_config):
        """Test loading brain template."""
        atlas = AtlasManager(test_config)

        # Load full template
        template = atlas.get_template()
        assert isinstance(template, nib.Nifti1Image)
        assert len(template.shape) == 3  # 3D image

        # Template should be cached
        template2 = atlas.get_template()
        assert template is template2  # Same object

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_template_with_slices(self, test_config):
        """Test loading template with slice extraction."""
        atlas = AtlasManager(test_config)

        # Load with DTI slice range (15-25 = 10 slices)
        dwi_template = atlas.get_template(modality='dwi')
        assert isinstance(dwi_template, nib.Nifti1Image)
        # Should have fewer slices than full template
        assert dwi_template.shape[2] <= 11  # ~10 slices

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_tissue_mask(self, test_config):
        """Test loading tissue masks."""
        atlas = AtlasManager(test_config)

        # Load GM mask
        gm_mask = atlas.get_tissue_mask('gm')
        assert isinstance(gm_mask, nib.Nifti1Image)

        # Load WM probability map
        wm_prob = atlas.get_tissue_mask('wm', probability=True)
        assert isinstance(wm_prob, nib.Nifti1Image)

        # Values should be different (binary vs probability)
        gm_data = gm_mask.get_fdata()
        wm_data = wm_prob.get_fdata()
        assert np.unique(gm_data).size <= 2  # Binary mask
        assert np.max(wm_data) <= 1.0  # Probability

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_brain_mask(self, test_config):
        """Test loading brain mask."""
        atlas = AtlasManager(test_config)

        mask = atlas.get_brain_mask()
        assert isinstance(mask, nib.Nifti1Image)

        # Should be binary
        mask_data = mask.get_fdata()
        unique_vals = np.unique(mask_data)
        assert len(unique_vals) <= 2

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_parcellation(self, test_config):
        """Test loading parcellation atlas."""
        atlas = AtlasManager(test_config)

        # Load anatomical parcellation
        parcellation = atlas.get_parcellation(atlas_type='anatomical')
        assert isinstance(parcellation, nib.Nifti1Image)

        # Should have discrete labels (may be stored as float or int)
        parc_data = parcellation.get_fdata()
        # Check that values are discrete (whole numbers)
        unique_vals = np.unique(parc_data)
        assert len(unique_vals) > 1  # Should have multiple ROIs
        # Verify values are integers (even if stored as float)
        assert np.allclose(unique_vals, np.round(unique_vals))

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_roi_labels(self, test_config):
        """Test loading ROI label definitions."""
        atlas = AtlasManager(test_config)

        roi_df = atlas.get_roi_labels()
        assert len(roi_df) > 0
        assert 'Labels' in roi_df.columns
        assert 'Region of interest' in roi_df.columns

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_get_roi_mask(self, test_config):
        """Test creating ROI-specific masks."""
        atlas = AtlasManager(test_config)

        # Get hippocampus mask
        hipp_mask = atlas.get_roi_mask('Hippocampus')
        assert isinstance(hipp_mask, nib.Nifti1Image)

        # Should be binary
        mask_data = hipp_mask.get_fdata()
        unique_vals = np.unique(mask_data)
        assert len(unique_vals) == 2  # 0 and 1
        assert np.sum(mask_data) > 0  # At least some voxels

    def test_get_slice_range(self, test_config):
        """Test getting configured slice range."""
        atlas = AtlasManager(test_config)

        # DTI should be 15-25
        start, end = atlas.get_slice_range('dwi')
        assert start == 15
        assert end == 25

        # Full anatomical should be 0 to -1
        start, end = atlas.get_slice_range('anat')
        assert start == 0
        assert end == -1

    def test_get_slice_range_invalid_modality(self, test_config):
        """Test error for invalid modality."""
        atlas = AtlasManager(test_config)

        with pytest.raises(ValueError, match="No slice definition"):
            atlas.get_slice_range('nonexistent')

    def test_cache_clearing(self, test_config):
        """Test clearing image cache."""
        atlas = AtlasManager(test_config)

        # Populate cache (if atlas available)
        if Path('/mnt/arborea/atlases/SIGMA').exists():
            _ = atlas.get_template()
            assert len(atlas.templates) > 0

            atlas.clear_cache()
            assert len(atlas.templates) == 0
            assert len(atlas.labels) == 0


class TestSliceExtraction:
    """Tests for slice extraction utilities."""

    def test_extract_slices_basic(self, mock_nifti_image):
        """Test basic slice extraction."""
        img = mock_nifti_image

        # Extract slices 10-20 (10 slices)
        extracted = extract_slices(img, 10, 20, axis=2)

        assert extracted.shape[0] == img.shape[0]
        assert extracted.shape[1] == img.shape[1]
        assert extracted.shape[2] == 10

    def test_extract_slices_to_end(self, mock_nifti_image):
        """Test extraction to end (-1)."""
        img = mock_nifti_image

        # Extract from 10 to end
        extracted = extract_slices(img, 10, -1, axis=2)

        expected_slices = img.shape[2] - 10
        assert extracted.shape[2] == expected_slices

    def test_extract_slices_affine_update(self, mock_nifti_image):
        """Test that affine matrix is correctly updated."""
        img = mock_nifti_image

        # Extract slices starting at 10
        extracted = extract_slices(img, 10, 20, axis=2)

        # Translation should be adjusted by 10 * voxel_size
        voxel_size = img.affine[2, 2]
        expected_offset = 10 * voxel_size

        # Check z-translation
        np.testing.assert_almost_equal(
            extracted.affine[2, 3] - img.affine[2, 3],
            expected_offset,
            decimal=5
        )

    def test_extract_slices_different_axes(self, mock_nifti_image):
        """Test extraction along different axes."""
        img = mock_nifti_image

        # Extract along x-axis
        extracted_x = extract_slices(img, 5, 15, axis=0)
        assert extracted_x.shape == (10, 40, 40)

        # Extract along y-axis
        extracted_y = extract_slices(img, 5, 15, axis=1)
        assert extracted_y.shape == (40, 10, 40)

        # Extract along z-axis
        extracted_z = extract_slices(img, 5, 15, axis=2)
        assert extracted_z.shape == (40, 40, 10)

    def test_extract_slices_invalid_range(self, mock_nifti_image):
        """Test error handling for invalid ranges."""
        img = mock_nifti_image

        # Start < 0
        with pytest.raises(ValueError, match="slice_start must be"):
            extract_slices(img, -1, 10)

        # End <= start
        with pytest.raises(ValueError, match="slice_end .* must be >"):
            extract_slices(img, 20, 10)

        # Start >= image size
        with pytest.raises(ValueError, match="exceeds image size"):
            extract_slices(img, 100, 110)

    def test_extract_slices_invalid_axis(self, mock_nifti_image):
        """Test error for invalid axis."""
        img = mock_nifti_image

        with pytest.raises(ValueError, match="axis must be"):
            extract_slices(img, 0, 10, axis=3)

    def test_get_slice_range_from_config(self, mock_nifti_image, test_config):
        """Test getting slice range from config."""
        img = mock_nifti_image

        start, end = get_slice_range(img, 'dwi', test_config)
        assert start == 15
        assert end == 25

    def test_get_slice_range_validation(self, mock_nifti_image, test_config):
        """Test that slice range is validated against image."""
        # Create small image (only 20 slices)
        small_img = nib.Nifti1Image(np.random.rand(40, 40, 20), np.eye(4))

        # Config specifies end=25, but image only has 20 slices
        # Should clip to 20 with warning
        start, end = get_slice_range(small_img, 'dwi', test_config)
        assert end == 20  # Clipped

    def test_extract_modality_slices(self, mock_nifti_image, test_config):
        """Test convenience function for modality extraction."""
        img = mock_nifti_image

        extracted = extract_modality_slices(img, 'dwi', test_config)
        assert extracted.shape[2] == 10  # 15-25 = 10 slices

    def test_match_slice_geometry(self, test_config):
        """Test matching slice geometry between images."""
        # Create source and target with different sizes
        source = nib.Nifti1Image(np.random.rand(40, 40, 30), np.eye(4))
        target = nib.Nifti1Image(np.random.rand(40, 40, 50), np.eye(4))

        # Match with config
        source_matched, target_matched = match_slice_geometry(
            source, target, modality='dwi', config=test_config
        )

        # Both should have same z-dimension (10 slices for dwi)
        assert source_matched.shape[2] == target_matched.shape[2]

    def test_match_slice_geometry_no_config(self):
        """Test geometry matching without config."""
        # Create images with different z-sizes
        source = nib.Nifti1Image(np.random.rand(40, 40, 20), np.eye(4))
        target = nib.Nifti1Image(np.random.rand(40, 40, 30), np.eye(4))

        # Should crop to minimum (20)
        source_matched, target_matched = match_slice_geometry(source, target)

        assert source_matched.shape[2] == 20
        assert target_matched.shape[2] == 20

    def test_get_slice_metadata(self, mock_nifti_image):
        """Test slice metadata generation."""
        img = mock_nifti_image

        metadata = get_slice_metadata(img, 15, 25, axis=2)

        assert metadata['slice_range'] == (15, 25)
        assert metadata['axis'] == 2
        assert metadata['original_shape'] == img.shape
        assert metadata['n_slices'] == 10
        assert 'description' in metadata


class TestAtlasIntegration:
    """Integration tests combining multiple atlas components."""

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_full_workflow_dwi(self, test_config):
        """Test complete workflow for DTI preprocessing."""
        atlas = AtlasManager(test_config)

        # Get hippocampus-focused template for DTI
        dwi_template = atlas.get_template(modality='dwi', masked=True)

        # Get tissue masks for the same region
        wm_mask = atlas.get_tissue_mask('wm', modality='dwi')

        # Z-dimension (slices) should match since both use same slice extraction
        assert dwi_template.shape[2] == wm_mask.shape[2]

        # Get slice range used
        start, end = atlas.get_slice_range('dwi')
        # Number of slices should match configured range
        assert dwi_template.shape[2] <= (end - start + 1)
        assert wm_mask.shape[2] <= (end - start + 1)

    @pytest.mark.skipif(
        not Path('/mnt/arborea/atlases/SIGMA').exists(),
        reason="SIGMA atlas not available"
    )
    def test_roi_extraction_workflow(self, test_config):
        """Test ROI extraction workflow."""
        atlas = AtlasManager(test_config)

        # Get hippocampus ROI
        hipp_mask = atlas.get_roi_mask('Hippocampus')

        # Get corresponding ROI labels
        roi_df = atlas.get_roi_labels()
        hipp_rois = roi_df[roi_df['Region of interest'].str.contains('Hippocampus', case=False)]

        # Should have found multiple hippocampal regions
        assert len(hipp_rois) > 0

        # Mask should contain some voxels
        assert np.sum(hipp_mask.get_fdata()) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
