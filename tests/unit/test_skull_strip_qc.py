"""
Unit tests for shared skull stripping QC utilities.

Tests with synthetic NIfTI data at varying slice counts (5, 9, 11, 41)
to cover MSME, func, DWI, and anat modalities respectively.
"""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from neurofaune.preprocess.qc.skull_strip_qc import (
    calculate_skull_strip_metrics,
    plot_slicesdir_mosaic,
    plot_mask_edge_triplanar,
    skull_strip_html_section,
)


def _make_synthetic_data(shape=(64, 64, 41), brain_fraction=0.3, seed=42):
    """Create synthetic original + mask data with a centered brain blob."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(10, 100, size=shape).astype(np.float32)

    # Create an ellipsoid brain mask centered in the volume
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx, ry, rz = shape[0] // 4, shape[1] // 4, shape[2] // 3
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    ellipsoid = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
    mask = (ellipsoid <= 1.0).astype(np.float32)

    # Boost brain signal
    brain_data = data * mask
    data[mask > 0] *= 2.0

    return data, brain_data, mask


class TestCalculateSkullStripMetrics:
    """Tests for calculate_skull_strip_metrics()."""

    def test_basic_metrics_keys(self):
        original, brain, mask = _make_synthetic_data()
        metrics = calculate_skull_strip_metrics(original, brain, mask)

        assert 'brain_volume_voxels' in metrics
        assert 'brain_to_total_ratio' in metrics
        assert 'snr_estimate' in metrics
        assert 'potential_over_stripping' in metrics
        assert 'potential_under_stripping' in metrics

    def test_brain_volume_positive(self):
        original, brain, mask = _make_synthetic_data()
        metrics = calculate_skull_strip_metrics(original, brain, mask)
        assert metrics['brain_volume_voxels'] > 0

    def test_volume_mm3_with_voxel_sizes(self):
        original, brain, mask = _make_synthetic_data()
        voxel_sizes = (0.15, 0.15, 0.5)
        metrics = calculate_skull_strip_metrics(
            original, brain, mask, voxel_sizes=voxel_sizes
        )
        assert 'brain_volume_mm3' in metrics
        expected = metrics['brain_volume_voxels'] * np.prod(voxel_sizes)
        assert abs(metrics['brain_volume_mm3'] - expected) < 1e-5

    def test_no_volume_mm3_without_voxel_sizes(self):
        original, brain, mask = _make_synthetic_data()
        metrics = calculate_skull_strip_metrics(original, brain, mask)
        assert 'brain_volume_mm3' not in metrics

    def test_skull_strip_info_merged(self):
        original, brain, mask = _make_synthetic_data()
        info = {'method': 'adaptive', 'extraction_ratio': 0.25}
        metrics = calculate_skull_strip_metrics(
            original, brain, mask, skull_strip_info=info
        )
        assert metrics['method'] == 'adaptive'
        assert metrics['extraction_ratio'] == 0.25

    def test_empty_mask_snr_zero(self):
        shape = (32, 32, 11)
        original = np.ones(shape, dtype=np.float32) * 50.0
        brain = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)

        metrics = calculate_skull_strip_metrics(original, brain, mask)
        assert metrics['snr_estimate'] == 0.0
        assert metrics['brain_volume_voxels'] == 0

    def test_quality_flags_normal(self):
        original, brain, mask = _make_synthetic_data()
        metrics = calculate_skull_strip_metrics(original, brain, mask)
        ratio = metrics['brain_to_total_ratio']
        # Our ellipsoid should give a reasonable ratio
        assert isinstance(metrics['potential_over_stripping'], bool)
        assert isinstance(metrics['potential_under_stripping'], bool)


class TestPlotSlicesdirMosaic:
    """Tests for plot_slicesdir_mosaic()."""

    @pytest.fixture
    def figures_dir(self, tmp_path):
        d = tmp_path / 'figures'
        d.mkdir()
        return d

    @pytest.mark.parametrize('n_slices,modality', [
        (5, 'msme'),
        (9, 'func'),
        (11, 'dwi'),
        (41, 'anat'),
    ])
    def test_mosaic_creates_png(self, figures_dir, n_slices, modality):
        original, _, mask = _make_synthetic_data(shape=(32, 32, n_slices))
        result = plot_slicesdir_mosaic(
            original, mask, 'sub-001', 'ses-p60', modality, figures_dir
        )
        assert result.exists()
        assert result.suffix == '.png'
        assert result.stat().st_size > 0

    def test_mosaic_max_slices_subsampling(self, figures_dir):
        """With >max_slices, should still produce a valid figure."""
        original, _, mask = _make_synthetic_data(shape=(32, 32, 80))
        result = plot_slicesdir_mosaic(
            original, mask, 'sub-001', 'ses-p60', 'anat', figures_dir,
            max_slices=20,
        )
        assert result.exists()

    def test_mosaic_single_slice(self, figures_dir):
        """Edge case: volume with a single slice."""
        original, _, mask = _make_synthetic_data(shape=(32, 32, 1))
        result = plot_slicesdir_mosaic(
            original, mask, 'sub-001', 'ses-p60', 'msme', figures_dir
        )
        assert result.exists()


class TestPlotMaskEdgeTriplanar:
    """Tests for plot_mask_edge_triplanar()."""

    @pytest.fixture
    def figures_dir(self, tmp_path):
        d = tmp_path / 'figures'
        d.mkdir()
        return d

    def test_triplanar_full_volume(self, figures_dir):
        """With >= 7 slices, should show all 3 orientations."""
        original, _, mask = _make_synthetic_data(shape=(32, 32, 41))
        result = plot_mask_edge_triplanar(
            original, mask, 'sub-001', 'ses-p60', 'anat', figures_dir
        )
        assert result.exists()
        assert result.suffix == '.png'

    def test_triplanar_partial_volume(self, figures_dir):
        """With < 7 slices, should show only axial."""
        original, _, mask = _make_synthetic_data(shape=(32, 32, 5))
        result = plot_mask_edge_triplanar(
            original, mask, 'sub-001', 'ses-p60', 'msme', figures_dir
        )
        assert result.exists()
        assert result.suffix == '.png'

    @pytest.mark.parametrize('modality', ['anat', 'dwi', 'func', 'msme'])
    def test_triplanar_all_modalities(self, figures_dir, modality):
        original, _, mask = _make_synthetic_data(shape=(32, 32, 11))
        result = plot_mask_edge_triplanar(
            original, mask, 'sub-001', 'ses-p60', modality, figures_dir
        )
        assert result.exists()


class TestSkullStripHtmlSection:
    """Tests for skull_strip_html_section()."""

    def _make_metrics_and_figures(self, tmp_path):
        metrics = {
            'brain_volume_voxels': 50000,
            'brain_to_total_ratio': 0.3,
            'snr_estimate': 8.5,
            'potential_over_stripping': False,
            'potential_under_stripping': False,
            'method': 'adaptive',
            'extraction_ratio': 0.25,
        }
        fig_dir = tmp_path / 'figures'
        fig_dir.mkdir(exist_ok=True)
        # Create dummy figure files
        fig1 = fig_dir / 'sub-001_ses-p60_skull_strip_mosaic.png'
        fig2 = fig_dir / 'sub-001_ses-p60_skull_strip_triplanar.png'
        fig1.write_bytes(b'\x89PNG\r\n')
        fig2.write_bytes(b'\x89PNG\r\n')
        return metrics, [fig1, fig2]

    def test_html_contains_section_div(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert '<div class="section">' in html
        assert '</div>' in html

    def test_html_contains_method(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert 'adaptive' in html

    def test_html_contains_extraction_ratio(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert '0.250' in html

    def test_html_contains_brain_volume(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert '50,000' in html

    def test_html_contains_snr(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert '8.50' in html

    def test_html_contains_figure_tags(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert '<img src="figures/' in html
        assert 'skull_strip_mosaic.png' in html
        assert 'skull_strip_triplanar.png' in html

    def test_html_quality_good(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        html = skull_strip_html_section(metrics, figures)
        assert 'GOOD' in html
        assert 'good' in html  # CSS class

    def test_html_quality_warning_over_stripping(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        metrics['potential_over_stripping'] = True
        html = skull_strip_html_section(metrics, figures)
        assert 'over-stripping' in html
        assert 'warning' in html

    def test_html_quality_warning_under_stripping(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        metrics['potential_under_stripping'] = True
        html = skull_strip_html_section(metrics, figures)
        assert 'under-stripping' in html

    def test_html_with_skull_strip_info_arg(self, tmp_path):
        metrics = {
            'brain_volume_voxels': 50000,
            'brain_to_total_ratio': 0.3,
            'snr_estimate': 8.5,
            'potential_over_stripping': False,
            'potential_under_stripping': False,
        }
        fig_dir = tmp_path / 'figures'
        fig_dir.mkdir(exist_ok=True)
        fig1 = fig_dir / 'sub-001_ses-p60_skull_strip_mosaic.png'
        fig1.write_bytes(b'\x89PNG\r\n')
        info = {'method': 'atropos_bet', 'extraction_ratio': 0.32}
        html = skull_strip_html_section(metrics, [fig1], skull_strip_info=info)
        assert 'atropos_bet' in html

    def test_html_with_volume_mm3(self, tmp_path):
        metrics, figures = self._make_metrics_and_figures(tmp_path)
        metrics['brain_volume_mm3'] = 562.5
        html = skull_strip_html_section(metrics, figures)
        assert '562.5 mm' in html
