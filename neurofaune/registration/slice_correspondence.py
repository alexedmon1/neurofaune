"""
Slice correspondence finder for partial-coverage modality registration.

This module implements a dual-approach system for determining where partial-coverage
modalities (DWI, fMRI) align with full-coverage T2w anatomical images:

1. Intensity-based matching: Correlates 2D slices from the partial volume with
   each slice in the T2w to find best matches.

2. Landmark detection: Identifies anatomical landmarks (ventricles, hippocampus)
   to validate and refine the intensity-based alignment.

The combination of both approaches provides robust, reproducible registration
even when header information is unavailable or unreliable.

Rodent MRI Challenge:
- T2w: 41 slices (328mm coverage at 8mm thickness)
- DWI: 11 slices (88mm coverage at 8mm thickness)
- fMRI: 9 slices (72mm coverage at 8mm thickness)
- All have affine origins at [0,0,0] - no positioning info in headers

Usage:
    from neurofaune.registration.slice_correspondence import SliceCorrespondenceFinder

    finder = SliceCorrespondenceFinder()
    result = finder.find_correspondence(
        partial_image=Path('dwi_b0.nii.gz'),
        full_image=Path('T2w.nii.gz'),
        modality='dwi'
    )
    print(f"Partial slices 0-10 correspond to T2w slices {result.start_slice}-{result.end_slice}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from scipy.signal import correlate2d
from scipy.stats import pearsonr


@dataclass
class SliceCorrespondenceResult:
    """Result of slice correspondence finding."""
    # Primary result
    start_slice: int  # First T2w slice corresponding to partial volume
    end_slice: int    # Last T2w slice corresponding to partial volume

    # Confidence metrics
    intensity_confidence: float  # 0-1, based on correlation quality
    landmark_confidence: float   # 0-1, based on landmark agreement
    combined_confidence: float   # 0-1, weighted combination

    # Detailed matching info
    slice_mapping: Dict[int, int]  # partial_slice_idx -> t2w_slice_idx
    intensity_correlations: List[float]  # Per-slice correlation values

    # Landmark info (if available)
    landmarks_found: Dict[str, Tuple[int, int]]  # landmark_name -> (partial_slice, t2w_slice)

    # Method info
    method_used: str  # 'intensity', 'landmark', 'combined'

    # Physical coordinate info (optional, for when voxel sizes differ)
    partial_slice_thickness: Optional[float] = None  # mm
    full_slice_thickness: Optional[float] = None     # mm
    physical_offset: Optional[float] = None          # mm from start of full volume

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_slice': self.start_slice,
            'end_slice': self.end_slice,
            'intensity_confidence': self.intensity_confidence,
            'landmark_confidence': self.landmark_confidence,
            'combined_confidence': self.combined_confidence,
            'slice_mapping': self.slice_mapping,
            'intensity_correlations': self.intensity_correlations,
            'landmarks_found': {k: list(v) for k, v in self.landmarks_found.items()},
            'method_used': self.method_used,
            'partial_slice_thickness': self.partial_slice_thickness,
            'full_slice_thickness': self.full_slice_thickness,
            'physical_offset': self.physical_offset,
        }


class IntensityMatcher:
    """
    Find slice correspondence using intensity correlation.

    Matches 2D slices from the partial-coverage volume to the full T2w
    by computing normalized cross-correlation for each potential alignment.
    """

    def __init__(
        self,
        search_margin: int = 5,
        min_correlation: float = 0.3,
        use_gradient: bool = True
    ):
        """
        Initialize intensity matcher.

        Parameters
        ----------
        search_margin : int
            Extra slices to search beyond expected range
        min_correlation : float
            Minimum correlation to consider a valid match
        use_gradient : bool
            Also match on gradient images (edge detection) for robustness
        """
        self.search_margin = search_margin
        self.min_correlation = min_correlation
        self.use_gradient = use_gradient

    def _preprocess_slice(self, slice_2d: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Preprocess a 2D slice for matching."""
        # Handle NaN values
        slice_2d = np.nan_to_num(slice_2d, nan=0.0)

        # Light smoothing to reduce noise
        smoothed = gaussian_filter(slice_2d.astype(float), sigma=1.0)

        if normalize:
            # Normalize to 0-1 range
            min_val = smoothed.min()
            max_val = smoothed.max()
            if max_val > min_val:
                smoothed = (smoothed - min_val) / (max_val - min_val)

        return smoothed

    def _compute_gradient(self, slice_2d: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude image."""
        gx = np.gradient(slice_2d, axis=0)
        gy = np.gradient(slice_2d, axis=1)
        return np.sqrt(gx**2 + gy**2)

    def _correlation_2d(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        handle_contrast_inversion: bool = True
    ) -> float:
        """
        Compute normalized correlation between two 2D images.

        Handles different image sizes by resizing to match.
        For modalities with inverted contrast (e.g., FA vs T2w),
        uses absolute value of correlation.

        Parameters
        ----------
        img1 : np.ndarray
            First image
        img2 : np.ndarray
            Second image
        handle_contrast_inversion : bool
            If True, return absolute value of correlation to handle
            modalities with inverted contrast (e.g., FA where WM is bright
            vs T2w where WM is dark)
        """
        # Flatten and compute correlation
        flat1 = img1.flatten()
        flat2 = img2.flatten()

        # Handle size mismatch by resizing
        if len(flat1) != len(flat2):
            # Resize img2 to match img1
            from scipy.ndimage import zoom
            zoom_factors = (img1.shape[0] / img2.shape[0],
                          img1.shape[1] / img2.shape[1])
            img2_resized = zoom(img2, zoom_factors, order=1)
            flat2 = img2_resized.flatten()

        # Create mask for non-zero regions (ignore background)
        mask = (flat1 > 0.05) | (flat2 > 0.05)

        if mask.sum() < 100:  # Need minimum pixels
            return 0.0

        try:
            corr, _ = pearsonr(flat1[mask], flat2[mask])
            if np.isnan(corr):
                return 0.0
            # For inverted-contrast modalities, use absolute correlation
            if handle_contrast_inversion:
                return abs(corr)
            return corr
        except Exception:
            return 0.0

    def _compute_physical_positions(
        self,
        n_slices: int,
        slice_thickness: float,
        start_offset: float = 0.0
    ) -> np.ndarray:
        """
        Compute physical Z positions for slice centers.

        Parameters
        ----------
        n_slices : int
            Number of slices
        slice_thickness : float
            Thickness of each slice in mm
        start_offset : float
            Starting position in mm

        Returns
        -------
        np.ndarray
            Physical Z position of each slice center
        """
        # Slice centers are at 0.5, 1.5, 2.5, ... times thickness
        return start_offset + (np.arange(n_slices) + 0.5) * slice_thickness

    def _find_closest_full_slice(
        self,
        partial_position: float,
        full_positions: np.ndarray
    ) -> int:
        """Find the index of the full slice closest to a physical position."""
        distances = np.abs(full_positions - partial_position)
        return int(np.argmin(distances))

    def match_slices(
        self,
        partial_data: np.ndarray,
        full_data: np.ndarray,
        partial_slice_thickness: Optional[float] = None,
        full_slice_thickness: Optional[float] = None
    ) -> Tuple[int, List[float], Dict[int, int]]:
        """
        Find best alignment of partial volume within full volume.

        Supports different slice thicknesses by matching based on physical
        position rather than assuming 1:1 slice correspondence.

        Parameters
        ----------
        partial_data : np.ndarray
            3D array of partial coverage volume (e.g., DWI b0)
        full_data : np.ndarray
            3D array of full coverage volume (T2w)
        partial_slice_thickness : float, optional
            Slice thickness of partial volume in mm. If None, assumes same as full.
        full_slice_thickness : float, optional
            Slice thickness of full volume in mm. If None, assumes 1.0 (index-based).

        Returns
        -------
        tuple
            (best_start_slice, correlation_list, slice_mapping)
        """
        n_partial = partial_data.shape[2]
        n_full = full_data.shape[2]

        # Handle slice thickness
        if full_slice_thickness is None:
            full_slice_thickness = 1.0  # Default to index-based
        if partial_slice_thickness is None:
            partial_slice_thickness = full_slice_thickness  # Assume same

        # Calculate physical coverage
        partial_coverage = n_partial * partial_slice_thickness
        full_coverage = n_full * full_slice_thickness

        if partial_coverage > full_coverage:
            raise ValueError(
                f"Partial volume ({partial_coverage:.1f}mm) has greater coverage than "
                f"full volume ({full_coverage:.1f}mm)"
            )

        # Physical positions of full volume slices (fixed)
        full_positions = self._compute_physical_positions(n_full, full_slice_thickness)

        # Search range in physical coordinates
        max_offset = full_coverage - partial_coverage
        # Convert search margin from slices to mm
        margin_mm = self.search_margin * full_slice_thickness

        # Test different starting offsets
        # Sample at half-slice resolution for finer positioning
        step_size = full_slice_thickness / 2
        test_offsets = np.arange(0, max_offset + step_size, step_size)

        position_scores = []

        for start_offset in test_offsets:
            # Physical positions of partial volume slices at this offset
            partial_positions = self._compute_physical_positions(
                n_partial, partial_slice_thickness, start_offset
            )

            slice_correlations = []
            slice_mapping = {}

            for partial_idx in range(n_partial):
                # Find closest full slice to this partial slice's position
                full_idx = self._find_closest_full_slice(
                    partial_positions[partial_idx], full_positions
                )

                slice_mapping[partial_idx] = full_idx

                # Get slices
                partial_slice = self._preprocess_slice(partial_data[:, :, partial_idx])
                full_slice = self._preprocess_slice(full_data[:, :, full_idx])

                # Intensity correlation
                intensity_corr = self._correlation_2d(partial_slice, full_slice)

                # Gradient correlation (if enabled)
                if self.use_gradient:
                    partial_grad = self._compute_gradient(partial_slice)
                    full_grad = self._compute_gradient(full_slice)
                    grad_corr = self._correlation_2d(partial_grad, full_grad)

                    # Combine intensity and gradient correlations
                    combined_corr = 0.7 * intensity_corr + 0.3 * grad_corr
                else:
                    combined_corr = intensity_corr

                slice_correlations.append(combined_corr)

            # Mean correlation for this position
            mean_corr = np.mean(slice_correlations)
            # Convert start_offset to equivalent slice index for output
            start_slice = int(round(start_offset / full_slice_thickness))
            position_scores.append((start_slice, mean_corr, slice_correlations, slice_mapping))

        # Find best position
        best_pos, best_score, best_correlations, best_mapping = max(
            position_scores, key=lambda x: x[1]
        )

        return best_pos, best_correlations, best_mapping


class LandmarkDetector:
    """
    Detect anatomical landmarks for slice correspondence validation.

    Currently detects:
    - Lateral ventricles (bright CSF on T2w, distinct shape)
    - Hippocampus region (based on position relative to ventricles)
    """

    def __init__(self):
        """Initialize landmark detector."""
        self.ventricle_threshold_percentile = 90
        self.min_ventricle_area = 50  # voxels

    def _detect_ventricles_in_slice(
        self,
        slice_2d: np.ndarray,
        is_t2w: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """
        Detect ventricles in a 2D slice.

        Returns
        -------
        tuple or None
            (center_x, center_y, area) if ventricles detected, else None
        """
        # Normalize
        slice_norm = slice_2d.astype(float)
        if slice_norm.max() > 0:
            slice_norm = slice_norm / slice_norm.max()
        else:
            # Empty slice - no ventricles possible
            return None

        # Check if there are non-zero pixels
        nonzero_pixels = slice_norm[slice_norm > 0]
        if len(nonzero_pixels) < self.min_ventricle_area:
            return None

        # Ventricles are bright on T2w
        if is_t2w:
            threshold = np.percentile(nonzero_pixels, self.ventricle_threshold_percentile)
            binary = slice_norm > threshold
        else:
            # For other modalities, look for similar intensity patterns
            threshold = np.percentile(nonzero_pixels, 85)
            binary = slice_norm > threshold

        # Fill holes and clean up
        binary = binary_fill_holes(binary)

        # Find connected components
        labeled, num_features = label(binary)

        if num_features == 0:
            return None

        # Find two largest connected components (bilateral ventricles)
        component_sizes = []
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            size = component_mask.sum()
            if size >= self.min_ventricle_area:
                # Get centroid
                coords = np.where(component_mask)
                center_y = np.mean(coords[0])
                center_x = np.mean(coords[1])
                component_sizes.append((i, size, center_x, center_y))

        if len(component_sizes) < 1:
            return None

        # Sort by size
        component_sizes.sort(key=lambda x: x[1], reverse=True)

        # Take largest (or mean of two if bilateral)
        if len(component_sizes) >= 2:
            # Average of two largest
            avg_x = (component_sizes[0][2] + component_sizes[1][2]) / 2
            avg_y = (component_sizes[0][3] + component_sizes[1][3]) / 2
            total_area = component_sizes[0][1] + component_sizes[1][1]
        else:
            avg_x = component_sizes[0][2]
            avg_y = component_sizes[0][3]
            total_area = component_sizes[0][1]

        return (avg_x, avg_y, total_area)

    def find_ventricle_profile(
        self,
        data_3d: np.ndarray,
        is_t2w: bool = True
    ) -> Tuple[List[int], List[float]]:
        """
        Find slices with largest ventricle representation.

        Returns
        -------
        tuple
            (slice_indices, ventricle_areas)
        """
        n_slices = data_3d.shape[2]

        slice_indices = []
        ventricle_areas = []

        for slice_idx in range(n_slices):
            result = self._detect_ventricles_in_slice(
                data_3d[:, :, slice_idx], is_t2w=is_t2w
            )
            if result is not None:
                _, _, area = result
                slice_indices.append(slice_idx)
                ventricle_areas.append(area)

        return slice_indices, ventricle_areas

    def find_peak_ventricle_slice(
        self,
        data_3d: np.ndarray,
        is_t2w: bool = True
    ) -> Optional[int]:
        """
        Find the slice with maximum ventricle visibility.

        This typically corresponds to the mid-brain level where
        lateral ventricles are most prominent.
        """
        slice_indices, areas = self.find_ventricle_profile(data_3d, is_t2w)

        if not areas:
            return None

        # Find peak
        max_idx = np.argmax(areas)
        return slice_indices[max_idx]

    def match_landmark_profiles(
        self,
        partial_data: np.ndarray,
        full_data: np.ndarray,
        partial_is_t2w: bool = False,
        full_is_t2w: bool = True
    ) -> Optional[Tuple[int, float, Dict[str, Tuple[int, int]]]]:
        """
        Match landmark profiles between partial and full volumes.

        Returns
        -------
        tuple or None
            (offset, confidence, landmarks_found)
        """
        # Get ventricle profiles for both volumes
        partial_slices, partial_areas = self.find_ventricle_profile(
            partial_data, is_t2w=partial_is_t2w
        )
        full_slices, full_areas = self.find_ventricle_profile(
            full_data, is_t2w=full_is_t2w
        )

        if not partial_areas or not full_areas:
            return None

        # Find peak slices
        partial_peak_local = np.argmax(partial_areas)
        partial_peak_slice = partial_slices[partial_peak_local]

        full_peak_local = np.argmax(full_areas)
        full_peak_slice = full_slices[full_peak_local]

        # Estimate offset based on peak alignment
        # If partial slice 3 has peak and full slice 15 has peak,
        # then partial volume likely starts at slice 15-3 = 12
        estimated_start = full_peak_slice - partial_peak_slice

        # Validate the offset makes sense
        n_partial = partial_data.shape[2]
        n_full = full_data.shape[2]

        if estimated_start < 0 or estimated_start + n_partial > n_full:
            # Offset is invalid, landmarks may be unreliable
            return None

        # Compute confidence based on profile similarity
        # Correlate the area profiles
        partial_profile = np.zeros(n_partial)
        for i, area in zip(partial_slices, partial_areas):
            partial_profile[i] = area

        full_profile_aligned = np.zeros(n_partial)
        for i in range(n_partial):
            full_idx = estimated_start + i
            if full_idx in full_slices:
                idx = full_slices.index(full_idx)
                full_profile_aligned[i] = full_areas[idx]

        # Normalize and correlate
        if partial_profile.sum() > 0 and full_profile_aligned.sum() > 0:
            partial_norm = partial_profile / partial_profile.max()
            full_norm = full_profile_aligned / full_profile_aligned.max()
            try:
                confidence, _ = pearsonr(partial_norm, full_norm)
                confidence = max(0, confidence)
            except Exception:
                confidence = 0.0
        else:
            confidence = 0.0

        landmarks_found = {
            'ventricle_peak': (partial_peak_slice, full_peak_slice)
        }

        return estimated_start, confidence, landmarks_found


class SliceCorrespondenceFinder:
    """
    Main class for finding slice correspondence using combined approaches.

    Combines intensity matching and landmark detection for robust results.
    """

    def __init__(
        self,
        intensity_weight: float = 0.6,
        landmark_weight: float = 0.4,
        min_confidence: float = 0.5
    ):
        """
        Initialize the correspondence finder.

        Parameters
        ----------
        intensity_weight : float
            Weight for intensity-based matching (0-1)
        landmark_weight : float
            Weight for landmark-based matching (0-1)
        min_confidence : float
            Minimum combined confidence to accept result
        """
        self.intensity_weight = intensity_weight
        self.landmark_weight = landmark_weight
        self.min_confidence = min_confidence

        self.intensity_matcher = IntensityMatcher()
        self.landmark_detector = LandmarkDetector()

    def find_correspondence(
        self,
        partial_image: Union[Path, np.ndarray, "nib.Nifti1Image"],
        full_image: Union[Path, np.ndarray, "nib.Nifti1Image"],
        modality: str = 'unknown',
        partial_is_t2w: bool = False,
        full_is_t2w: bool = True,
        partial_slice_thickness: Optional[float] = None,
        full_slice_thickness: Optional[float] = None
    ) -> SliceCorrespondenceResult:
        """
        Find correspondence between partial and full coverage volumes.

        Parameters
        ----------
        partial_image : Path, np.ndarray, or Nifti1Image
            Partial coverage image (e.g., DWI b0, mean fMRI)
        full_image : Path, np.ndarray, or Nifti1Image
            Full coverage reference image (T2w)
        modality : str
            Modality name for logging ('dwi', 'func', etc.)
        partial_is_t2w : bool
            Whether partial image is T2w-like contrast
        full_is_t2w : bool
            Whether full image is T2w contrast
        partial_slice_thickness : float, optional
            Slice thickness of partial volume in mm. If None, extracted from header
            or assumed same as full volume.
        full_slice_thickness : float, optional
            Slice thickness of full volume in mm. If None, extracted from header.

        Returns
        -------
        SliceCorrespondenceResult
            Detailed correspondence result with confidence metrics
        """
        # Load images if paths provided, extracting voxel sizes from headers
        if isinstance(partial_image, Path):
            partial_img = nib.load(partial_image)
            partial_data = partial_img.get_fdata()
            if partial_slice_thickness is None:
                partial_slice_thickness = float(partial_img.header.get_zooms()[2])
        elif hasattr(partial_image, 'get_fdata'):
            # It's a NIfTI image object
            partial_data = partial_image.get_fdata()
            if partial_slice_thickness is None:
                partial_slice_thickness = float(partial_image.header.get_zooms()[2])
        else:
            partial_data = partial_image

        if isinstance(full_image, Path):
            full_img = nib.load(full_image)
            full_data = full_img.get_fdata()
            if full_slice_thickness is None:
                full_slice_thickness = float(full_img.header.get_zooms()[2])
        elif hasattr(full_image, 'get_fdata'):
            # It's a NIfTI image object
            full_data = full_image.get_fdata()
            if full_slice_thickness is None:
                full_slice_thickness = float(full_image.header.get_zooms()[2])
        else:
            full_data = full_image

        # Handle 4D data (take first volume)
        if partial_data.ndim == 4:
            partial_data = partial_data[:, :, :, 0]
        if full_data.ndim == 4:
            full_data = full_data[:, :, :, 0]

        n_partial = partial_data.shape[2]
        n_full = full_data.shape[2]

        # Calculate physical coverage
        partial_coverage = n_partial * partial_slice_thickness if partial_slice_thickness else None
        full_coverage = n_full * full_slice_thickness if full_slice_thickness else None

        print(f"\nFinding slice correspondence for {modality}:")
        print(f"  Partial: {partial_data.shape} ({n_partial} slices)")
        if partial_slice_thickness:
            print(f"    Slice thickness: {partial_slice_thickness:.2f}mm, Coverage: {partial_coverage:.1f}mm")
        print(f"  Full: {full_data.shape} ({n_full} slices)")
        if full_slice_thickness:
            print(f"    Slice thickness: {full_slice_thickness:.2f}mm, Coverage: {full_coverage:.1f}mm")

        # Method 1: Intensity-based matching (with physical coordinates)
        intensity_start, intensity_corrs, intensity_mapping = \
            self.intensity_matcher.match_slices(
                partial_data, full_data,
                partial_slice_thickness=partial_slice_thickness,
                full_slice_thickness=full_slice_thickness
            )

        intensity_confidence = np.mean(intensity_corrs) if intensity_corrs else 0.0
        print(f"  Intensity matching: start={intensity_start}, confidence={intensity_confidence:.3f}")

        # Method 2: Landmark-based matching
        landmark_result = self.landmark_detector.match_landmark_profiles(
            partial_data, full_data,
            partial_is_t2w=partial_is_t2w,
            full_is_t2w=full_is_t2w
        )

        if landmark_result is not None:
            landmark_start, landmark_confidence, landmarks_found = landmark_result
            print(f"  Landmark matching: start={landmark_start}, confidence={landmark_confidence:.3f}")
        else:
            landmark_start = None
            landmark_confidence = 0.0
            landmarks_found = {}
            print(f"  Landmark matching: no landmarks detected")

        # Combine results
        if landmark_start is not None:
            # Check if methods agree
            agreement = abs(intensity_start - landmark_start) <= 2  # Within 2 slices

            if agreement:
                # Methods agree - high confidence
                combined_start = intensity_start
                combined_confidence = (
                    self.intensity_weight * intensity_confidence +
                    self.landmark_weight * landmark_confidence
                )
                method_used = 'combined'
                print(f"  Methods AGREE (within 2 slices)")
            else:
                # Methods disagree - use higher confidence method
                if intensity_confidence > landmark_confidence:
                    combined_start = intensity_start
                    combined_confidence = intensity_confidence * 0.8  # Penalize disagreement
                    method_used = 'intensity'
                    print(f"  Methods DISAGREE - using intensity (higher confidence)")
                else:
                    combined_start = landmark_start
                    combined_confidence = landmark_confidence * 0.8
                    method_used = 'landmark'
                    print(f"  Methods DISAGREE - using landmark (higher confidence)")
        else:
            # Landmark detection failed - rely on intensity
            combined_start = intensity_start
            combined_confidence = intensity_confidence * 0.9  # Slight penalty for single method
            method_used = 'intensity'
            print(f"  Using intensity only (landmarks unavailable)")

        # Build final slice mapping
        # Use intensity mapping when methods agree or intensity is used,
        # otherwise build simple sequential mapping from combined_start
        if method_used in ('combined', 'intensity') and intensity_mapping:
            final_mapping = intensity_mapping
        else:
            final_mapping = {
                i: combined_start + i
                for i in range(n_partial)
            }

        # Calculate physical offset (mm from start of full volume)
        physical_offset = None
        if full_slice_thickness is not None:
            physical_offset = combined_start * full_slice_thickness

        result = SliceCorrespondenceResult(
            start_slice=combined_start,
            end_slice=combined_start + n_partial - 1,
            intensity_confidence=intensity_confidence,
            landmark_confidence=landmark_confidence,
            combined_confidence=combined_confidence,
            slice_mapping=final_mapping,
            intensity_correlations=intensity_corrs,
            landmarks_found=landmarks_found,
            method_used=method_used,
            partial_slice_thickness=partial_slice_thickness,
            full_slice_thickness=full_slice_thickness,
            physical_offset=physical_offset
        )

        print(f"  RESULT: slices {result.start_slice}-{result.end_slice}, "
              f"confidence={result.combined_confidence:.3f}")

        return result

    def validate_correspondence(
        self,
        result: SliceCorrespondenceResult,
        partial_data: np.ndarray,
        full_data: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validate a correspondence result.

        Returns
        -------
        tuple
            (is_valid, message)
        """
        # Check confidence threshold
        if result.combined_confidence < self.min_confidence:
            return False, f"Low confidence ({result.combined_confidence:.3f} < {self.min_confidence})"

        # Check slice range is valid
        n_partial = partial_data.shape[2]
        n_full = full_data.shape[2]

        if result.start_slice < 0:
            return False, f"Invalid start slice ({result.start_slice} < 0)"

        if result.end_slice >= n_full:
            return False, f"Invalid end slice ({result.end_slice} >= {n_full})"

        # Check correlations are reasonable
        mean_corr = np.mean(result.intensity_correlations)
        if mean_corr < 0.2:
            return False, f"Poor correlation ({mean_corr:.3f})"

        return True, "Valid"


def find_slice_correspondence(
    partial_image: Union[str, Path],
    full_image: Union[str, Path],
    modality: str = 'unknown'
) -> SliceCorrespondenceResult:
    """
    Convenience function to find slice correspondence.

    Parameters
    ----------
    partial_image : str or Path
        Path to partial coverage image
    full_image : str or Path
        Path to full coverage reference image (T2w)
    modality : str
        Modality name ('dwi', 'func', etc.)

    Returns
    -------
    SliceCorrespondenceResult
        Correspondence result

    Examples
    --------
    >>> result = find_slice_correspondence(
    ...     partial_image='sub-001_dwi_b0.nii.gz',
    ...     full_image='sub-001_T2w.nii.gz',
    ...     modality='dwi'
    ... )
    >>> print(f"DWI slices 0-10 -> T2w slices {result.start_slice}-{result.end_slice}")
    """
    finder = SliceCorrespondenceFinder()
    return finder.find_correspondence(
        partial_image=Path(partial_image),
        full_image=Path(full_image),
        modality=modality
    )
