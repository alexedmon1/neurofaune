"""
Unit tests for slice correspondence module.

Tests the dual-approach (intensity + landmark) slice matching system
for registering partial-coverage modalities to full T2w.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from neurofaune.registration.slice_correspondence import (
    SliceCorrespondenceFinder,
    SliceCorrespondenceResult,
    IntensityMatcher,
    LandmarkDetector,
    find_slice_correspondence,
)


class TestIntensityMatcher:
    """Tests for IntensityMatcher class."""

    def test_preprocess_slice_normalization(self):
        """Test that slice preprocessing normalizes values to 0-1."""
        matcher = IntensityMatcher()

        # Create test slice with range 100-1000
        test_slice = np.random.rand(64, 64) * 900 + 100

        processed = matcher._preprocess_slice(test_slice, normalize=True)

        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_preprocess_slice_handles_nan(self):
        """Test that preprocessing handles NaN values."""
        matcher = IntensityMatcher()

        test_slice = np.random.rand(64, 64) * 100
        test_slice[10:20, 10:20] = np.nan

        processed = matcher._preprocess_slice(test_slice)

        assert not np.any(np.isnan(processed))

    def test_compute_gradient(self):
        """Test gradient computation produces expected output shape."""
        matcher = IntensityMatcher()

        test_slice = np.random.rand(64, 64)
        gradient = matcher._compute_gradient(test_slice)

        assert gradient.shape == test_slice.shape
        assert gradient.min() >= 0.0  # Gradient magnitude is non-negative

    def test_correlation_2d_identical_images(self):
        """Test that identical images have correlation ~1."""
        matcher = IntensityMatcher()

        img = np.random.rand(64, 64) * 0.5 + 0.25  # Values in 0.25-0.75

        corr = matcher._correlation_2d(img, img)

        assert corr > 0.99  # Should be very close to 1

    def test_correlation_2d_different_images(self):
        """Test correlation between different images."""
        matcher = IntensityMatcher()

        img1 = np.random.rand(64, 64)
        img2 = np.random.rand(64, 64)

        corr = matcher._correlation_2d(img1, img2)

        # Random images should have low correlation
        assert -0.5 < corr < 0.5

    def test_match_slices_finds_correct_offset(self):
        """Test that matching finds the correct slice offset."""
        matcher = IntensityMatcher()

        # Create full volume with 41 slices
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41) * 0.5 + 0.25

        # Create partial volume as a subset (11 slices starting at slice 15)
        true_offset = 15
        n_partial = 11
        partial_data = full_data[:, :, true_offset:true_offset+n_partial].copy()

        # Add small noise
        partial_data += np.random.rand(*partial_data.shape) * 0.05

        best_start, correlations, mapping = matcher.match_slices(partial_data, full_data)

        # Should find approximately the correct offset
        assert abs(best_start - true_offset) <= 2
        assert len(correlations) == n_partial
        assert len(mapping) == n_partial

    def test_match_slices_high_correlation_for_correct_match(self):
        """Test that correct match produces high correlation."""
        matcher = IntensityMatcher()

        # Create matching volumes
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)

        true_offset = 15
        partial_data = full_data[:, :, true_offset:true_offset+11].copy()

        _, correlations, _ = matcher.match_slices(partial_data, full_data)

        # Correlations should be high for correct match
        assert np.mean(correlations) > 0.7

    def test_compute_physical_positions(self):
        """Test physical position computation."""
        matcher = IntensityMatcher()

        # 5 slices at 8mm thickness, starting at 0
        positions = matcher._compute_physical_positions(5, 8.0, 0.0)

        # Slice centers should be at 4, 12, 20, 28, 36 mm
        expected = np.array([4.0, 12.0, 20.0, 28.0, 36.0])
        np.testing.assert_array_almost_equal(positions, expected)

    def test_compute_physical_positions_with_offset(self):
        """Test physical position computation with starting offset."""
        matcher = IntensityMatcher()

        # 3 slices at 2mm thickness, starting at 10mm
        positions = matcher._compute_physical_positions(3, 2.0, 10.0)

        # Slice centers should be at 11, 13, 15 mm
        expected = np.array([11.0, 13.0, 15.0])
        np.testing.assert_array_almost_equal(positions, expected)

    def test_find_closest_full_slice(self):
        """Test finding closest slice by physical position."""
        matcher = IntensityMatcher()

        # Full slices at positions 0, 8, 16, 24, 32 mm (8mm thickness, centers)
        full_positions = np.array([4.0, 12.0, 20.0, 28.0, 36.0])

        # Find closest to position 13 mm -> should be slice 1 (at 12mm)
        closest = matcher._find_closest_full_slice(13.0, full_positions)
        assert closest == 1

        # Find closest to position 25 mm -> should be slice 3 (at 28mm)
        closest = matcher._find_closest_full_slice(25.0, full_positions)
        assert closest == 3

    def test_match_slices_with_different_thicknesses(self):
        """Test matching with different slice thicknesses."""
        matcher = IntensityMatcher()

        # Create full volume: 41 slices at 8mm = 328mm coverage
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)

        # Create partial volume: 11 slices at 8mm = 88mm coverage
        # Starting at slice 15 of full (offset = 120mm)
        true_offset = 15
        partial_data = full_data[:, :, true_offset:true_offset+11].copy()

        # Test with explicit slice thicknesses
        best_start, correlations, mapping = matcher.match_slices(
            partial_data, full_data,
            partial_slice_thickness=8.0,
            full_slice_thickness=8.0
        )

        # Should find approximately correct offset
        assert abs(best_start - true_offset) <= 2
        assert len(mapping) == 11


class TestLandmarkDetector:
    """Tests for LandmarkDetector class."""

    def test_detect_ventricles_returns_none_for_empty(self):
        """Test that empty slice returns None."""
        detector = LandmarkDetector()

        empty_slice = np.zeros((64, 64))
        result = detector._detect_ventricles_in_slice(empty_slice)

        assert result is None

    def test_detect_ventricles_finds_bright_regions(self):
        """Test that bright regions are detected as potential ventricles."""
        detector = LandmarkDetector()

        # Create slice with bright central region (simulating ventricles)
        test_slice = np.random.rand(64, 64) * 0.3
        test_slice[25:35, 20:30] = 0.9  # Bright left ventricle
        test_slice[25:35, 35:45] = 0.9  # Bright right ventricle

        result = detector._detect_ventricles_in_slice(test_slice)

        assert result is not None
        center_x, center_y, area = result
        # Center should be roughly in the middle
        assert 25 < center_x < 40
        assert 25 < center_y < 40

    def test_find_ventricle_profile(self):
        """Test ventricle profile detection across volume."""
        detector = LandmarkDetector()

        # Create volume with varying "ventricle" sizes
        data_3d = np.random.rand(64, 64, 20) * 0.3

        # Add bright regions that get larger then smaller (like real ventricles)
        for z in range(5, 15):
            size = 5 + abs(z - 10) * -1 + 5  # Peaks at z=10
            size = max(3, size)
            c = 32  # center
            data_3d[c-size:c+size, c-size:c+size, z] = 0.9

        slice_indices, areas = detector.find_ventricle_profile(data_3d)

        assert len(slice_indices) > 0
        assert len(areas) == len(slice_indices)

    def test_find_peak_ventricle_slice(self):
        """Test finding the slice with maximum ventricle visibility."""
        detector = LandmarkDetector()

        # Create volume with peak ventricle at slice 10
        data_3d = np.random.rand(64, 64, 20) * 0.2

        # Add bright regions with peak at z=10
        for z in range(5, 15):
            size = 3 + (5 - abs(z - 10))
            c = 32
            data_3d[c-size:c+size, c-size:c+size, z] = 0.9

        peak_slice = detector.find_peak_ventricle_slice(data_3d)

        # Should find peak around slice 10
        assert peak_slice is not None
        assert 8 <= peak_slice <= 12


class TestSliceCorrespondenceFinder:
    """Tests for SliceCorrespondenceFinder class."""

    def test_initialization(self):
        """Test finder initialization with custom weights."""
        finder = SliceCorrespondenceFinder(
            intensity_weight=0.7,
            landmark_weight=0.3,
            min_confidence=0.4
        )

        assert finder.intensity_weight == 0.7
        assert finder.landmark_weight == 0.3
        assert finder.min_confidence == 0.4

    def test_find_correspondence_with_arrays(self):
        """Test finding correspondence with numpy arrays."""
        finder = SliceCorrespondenceFinder()

        # Create matching volumes
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)
        partial_data = full_data[:, :, 15:26].copy()

        result = finder.find_correspondence(
            partial_image=partial_data,
            full_image=full_data,
            modality='test'
        )

        assert isinstance(result, SliceCorrespondenceResult)
        assert result.start_slice >= 0
        assert result.end_slice < 41
        assert len(result.slice_mapping) == 11

    def test_find_correspondence_handles_4d_data(self):
        """Test that 4D data is handled correctly (uses first volume)."""
        finder = SliceCorrespondenceFinder()

        # Create 4D partial data (e.g., fMRI with multiple timepoints)
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)
        partial_data_4d = np.random.rand(64, 64, 11, 10)
        partial_data_4d[:, :, :, 0] = full_data[:, :, 15:26]

        result = finder.find_correspondence(
            partial_image=partial_data_4d,
            full_image=full_data,
            modality='func'
        )

        assert isinstance(result, SliceCorrespondenceResult)
        assert result.start_slice >= 0

    def test_result_to_dict(self):
        """Test that result can be serialized to dictionary."""
        result = SliceCorrespondenceResult(
            start_slice=15,
            end_slice=25,
            intensity_confidence=0.8,
            landmark_confidence=0.7,
            combined_confidence=0.75,
            slice_mapping={0: 15, 1: 16, 2: 17},
            intensity_correlations=[0.8, 0.82, 0.79],
            landmarks_found={'ventricle_peak': (5, 20)},
            method_used='combined'
        )

        result_dict = result.to_dict()

        assert result_dict['start_slice'] == 15
        assert result_dict['end_slice'] == 25
        assert result_dict['combined_confidence'] == 0.75
        assert 'slice_mapping' in result_dict
        assert 'landmarks_found' in result_dict

    def test_result_includes_physical_coordinates(self):
        """Test that result includes physical coordinate information."""
        result = SliceCorrespondenceResult(
            start_slice=15,
            end_slice=25,
            intensity_confidence=0.8,
            landmark_confidence=0.7,
            combined_confidence=0.75,
            slice_mapping={0: 15, 1: 16, 2: 17},
            intensity_correlations=[0.8, 0.82, 0.79],
            landmarks_found={'ventricle_peak': (5, 20)},
            method_used='combined',
            partial_slice_thickness=8.0,
            full_slice_thickness=8.0,
            physical_offset=120.0  # 15 slices * 8mm
        )

        assert result.partial_slice_thickness == 8.0
        assert result.full_slice_thickness == 8.0
        assert result.physical_offset == 120.0

        # Check serialization includes physical fields
        result_dict = result.to_dict()
        assert result_dict['partial_slice_thickness'] == 8.0
        assert result_dict['full_slice_thickness'] == 8.0
        assert result_dict['physical_offset'] == 120.0

    def test_find_correspondence_populates_physical_info(self):
        """Test that find_correspondence populates physical coordinate info."""
        finder = SliceCorrespondenceFinder()

        # Create matching volumes
        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)
        partial_data = full_data[:, :, 15:26].copy()

        result = finder.find_correspondence(
            partial_image=partial_data,
            full_image=full_data,
            modality='test',
            partial_slice_thickness=8.0,
            full_slice_thickness=8.0
        )

        # Should have physical coordinate info populated
        assert result.partial_slice_thickness == 8.0
        assert result.full_slice_thickness == 8.0
        assert result.physical_offset is not None
        # physical_offset should be start_slice * full_slice_thickness
        assert result.physical_offset == result.start_slice * 8.0

    def test_validate_correspondence_low_confidence(self):
        """Test validation rejects low confidence results."""
        finder = SliceCorrespondenceFinder(min_confidence=0.5)

        result = SliceCorrespondenceResult(
            start_slice=15,
            end_slice=25,
            intensity_confidence=0.3,
            landmark_confidence=0.2,
            combined_confidence=0.25,
            slice_mapping={},
            intensity_correlations=[0.3],
            landmarks_found={},
            method_used='intensity'
        )

        partial_data = np.random.rand(64, 64, 11)
        full_data = np.random.rand(64, 64, 41)

        is_valid, message = finder.validate_correspondence(result, partial_data, full_data)

        assert not is_valid
        assert 'confidence' in message.lower()

    def test_validate_correspondence_invalid_range(self):
        """Test validation rejects invalid slice ranges."""
        finder = SliceCorrespondenceFinder()

        result = SliceCorrespondenceResult(
            start_slice=-5,  # Invalid negative start
            end_slice=5,
            intensity_confidence=0.8,
            landmark_confidence=0.8,
            combined_confidence=0.8,
            slice_mapping={},
            intensity_correlations=[0.8],
            landmarks_found={},
            method_used='combined'
        )

        partial_data = np.random.rand(64, 64, 11)
        full_data = np.random.rand(64, 64, 41)

        is_valid, message = finder.validate_correspondence(result, partial_data, full_data)

        assert not is_valid
        assert 'start slice' in message.lower()


class TestConvenienceFunction:
    """Tests for the find_slice_correspondence convenience function."""

    @patch('neurofaune.registration.slice_correspondence.nib.load')
    def test_find_slice_correspondence_with_paths(self, mock_load):
        """Test convenience function loads files correctly."""
        # Mock nibabel loading
        partial_data = np.random.rand(64, 64, 11)
        full_data = np.random.rand(64, 64, 41)

        mock_partial_img = MagicMock()
        mock_partial_img.get_fdata.return_value = partial_data

        mock_full_img = MagicMock()
        mock_full_img.get_fdata.return_value = full_data

        def side_effect(path):
            if 'partial' in str(path):
                return mock_partial_img
            return mock_full_img

        mock_load.side_effect = side_effect

        result = find_slice_correspondence(
            partial_image=Path('partial.nii.gz'),
            full_image=Path('full.nii.gz'),
            modality='test'
        )

        assert isinstance(result, SliceCorrespondenceResult)
        assert mock_load.call_count == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_partial_larger_than_full_raises(self):
        """Test error when partial volume is larger than full."""
        matcher = IntensityMatcher()

        partial_data = np.random.rand(64, 64, 50)  # 50 slices
        full_data = np.random.rand(64, 64, 30)     # Only 30 slices

        with pytest.raises(ValueError, match="greater coverage"):
            matcher.match_slices(partial_data, full_data)

    def test_zero_intensity_slice(self):
        """Test handling of zero-intensity slices."""
        matcher = IntensityMatcher()

        # Create volumes where partial has a zero slice
        partial_data = np.random.rand(64, 64, 11)
        partial_data[:, :, 5] = 0  # One zero slice
        full_data = np.random.rand(64, 64, 41)

        # Should not crash
        best_start, correlations, mapping = matcher.match_slices(partial_data, full_data)

        assert best_start >= 0
        assert len(correlations) == 11

    def test_single_slice_partial(self):
        """Test handling of single-slice partial volume."""
        finder = SliceCorrespondenceFinder()

        np.random.seed(42)
        full_data = np.random.rand(64, 64, 41)
        partial_data = full_data[:, :, 20:21].copy()  # Single slice

        result = finder.find_correspondence(
            partial_image=partial_data,
            full_image=full_data,
            modality='single'
        )

        assert result.start_slice == result.end_slice
        assert len(result.slice_mapping) == 1
