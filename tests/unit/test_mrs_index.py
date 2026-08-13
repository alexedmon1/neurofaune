#!/usr/bin/env python3
"""
Unit tests for the study-level MRS QC index.

Built against a synthetic MRS output tree, so they need neither a study on disk
nor FSL.
"""

import json

import pytest

from neurofaune.preprocess.qc.mrs import QC_THRESHOLDS, generate_mrs_index


def make_session(root, subject, session, metrics=None, meta=None, reports=()):
    """Write one session's QC metrics, metadata and (empty) report files."""
    qc_dir = root / 'qc' / subject / session
    figures = qc_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    payload = {
        'reference_peak': 'NAA', 'snr': 25.0, 'fwhm_hz': 8.0,
        'n_metabolites': 19, 'n_metabolites_reliable': 17,
        'voxel_coverage': 1.0, 'snr_pass': True, 'fwhm_pass': True,
        'coverage_pass': True, 'overall_pass': True,
    }
    payload.update(metrics or {})
    (qc_dir / f'{subject}_{session}_mrs-qc.json').write_text(json.dumps(payload))

    session_dir = root / subject / session
    session_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        'fitter': 'fsl_mrs', 'source': 'rawdata', 'n_averages': 256,
        'water_removed': False,
        'tissue_fractions': {'GM': 0.7, 'WM': 0.15, 'CSF': 0.15, 'measured': True},
    }
    meta_payload.update(meta or {})
    (session_dir / f'{subject}_{session}_mrs.json').write_text(json.dumps(meta_payload))

    for name in reports:
        target = (figures / name if name.endswith('.png') or 'qc.html' in name
                  else session_dir / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('x')
    return qc_dir


@pytest.fixture
def study(tmp_path):
    root = tmp_path / 'mrs'
    make_session(root, 'sub-1Y', 'ses-1',
                 reports=('sub-1Y_ses-1_mrs-qc.html',
                          'sub-1Y_ses-1_voxel-placement.png',
                          'fit/report.html'))
    make_session(root, 'sub-1Y', 'ses-2',
                 metrics={'snr': 6.0, 'snr_pass': False, 'overall_pass': False})
    make_session(root, 'sub-2Y', 'ses-1',
                 meta={'water_removed': True,
                       'tissue_fractions': {'GM': 0.6, 'WM': 0.2, 'CSF': 0.2,
                                            'measured': False}})
    return root


class TestIndex:
    def test_writes_a_row_per_session(self, study):
        out = generate_mrs_index(study, study_name='test')
        html = out.read_text()
        assert out.name == 'index.html'
        assert html.count('<tr><td class="sess">') == 3
        for subject in ('sub-1Y', 'sub-2Y'):
            assert subject in html

    def test_summary_counts(self, study):
        html = generate_mrs_index(study).read_text()
        # 3 sessions, 2 subjects, 2 passing, 2 with measured fractions.
        assert '>3</div>' in html
        assert '2 subjects' in html
        assert '2/3' in html

    def test_flags_a_failing_metric(self, study):
        html = generate_mrs_index(study).read_text()
        assert '>REVIEW<' in html
        # The failing SNR cell carries the flag class, not just the status.
        assert 'class="flag">6.0' in html

    def test_notes_assumed_fractions_and_retries(self, study):
        html = generate_mrs_index(study).read_text()
        assert 'assumed tissue fractions' in html
        assert 'HLSVD retry' in html

    def test_links_are_relative_to_the_root(self, study):
        html = generate_mrs_index(study).read_text()
        # Portability: no absolute paths, and the fsl_mrs report is linked.
        assert f'href="{study}' not in html
        assert 'href="sub-1Y/ses-1/fit/report.html"' in html
        assert 'href="qc/sub-1Y/ses-1/figures/sub-1Y_ses-1_voxel-placement.png"' in html

    def test_thresholds_match_the_per_session_reports(self, study):
        html = generate_mrs_index(study).read_text()
        assert f"{QC_THRESHOLDS['min_snr']:.0f}" in html
        assert f"{QC_THRESHOLDS['max_fwhm_hz']:.0f}" in html

    def test_missing_metric_renders_blank_not_zero(self, tmp_path):
        root = tmp_path / 'mrs'
        make_session(root, 'sub-3Y', 'ses-1', metrics={'snr': float('nan')})
        html = generate_mrs_index(root).read_text()
        assert 'class="none">-' in html

    def test_no_qc_returns_none(self, tmp_path):
        assert generate_mrs_index(tmp_path / 'absent') is None
        empty = tmp_path / 'mrs'
        (empty / 'qc').mkdir(parents=True)
        assert generate_mrs_index(empty) is None

    def test_regenerating_is_idempotent(self, study):
        first = generate_mrs_index(study, study_name='test').read_text()
        second = generate_mrs_index(study, study_name='test').read_text()
        assert first == second
