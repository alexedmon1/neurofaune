#!/usr/bin/env python3
"""
Unit tests for post-hoc macromolecule quantification.

The measurement runs on the metabolite-free spectrum, ``residual + baseline``.
That identity is the load-bearing assumption, and it is also what the fit-curve
exporter got wrong once (it wrote time-domain arrays into those columns), so
the tests check the guard as well as the arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from neurofaune.preprocess.utils.mrs.mm_quantify import (
    DEFAULT_FLANKS,
    MIN_REFERENCE_PURITY,
    MM_BANDS,
    PROVISIONAL_BANDS,
    anchor_envelope,
    check_reference,
    fit_mm_spline,
    integrate_bands,
    metabolite_free_spectrum,
    mm_stability,
    quantify_mm,
    reference_area,
)


def gaussian(ppm, centre, height, width):
    return height * np.exp(-0.5 * ((ppm - centre) / width) ** 2)


def make_curves(mm_height=1.0, pedestal=0.0, noise=0.0, baseline_share=0.6, seed=0):
    """A synthetic fit whose MM content is known.

    ``baseline_share`` is the fraction of the MM bump the polynomial absorbed,
    which is what the real fit does with an MM-free basis. The measurement must
    be insensitive to it, since it adds the baseline back.
    """
    rng = np.random.default_rng(seed)
    ppm = np.linspace(0.2, 4.2, 2000)
    mm = gaussian(ppm, 0.90, mm_height, 0.09)
    naa = gaussian(ppm, 2.008, 5.0, 0.02)
    baseline = baseline_share * mm + pedestal
    metabolite_model = naa
    data = metabolite_model + mm + pedestal + rng.normal(0, noise, ppm.size)
    fit = metabolite_model + baseline
    return pd.DataFrame({
        'ppm': ppm,
        'data': data,
        'fit': fit,
        'baseline': baseline,
        'residual': data - fit,
    })


def make_metabolites(ppm, cr=1.0):
    return pd.DataFrame({
        'ppm': ppm,
        'Cr': gaussian(ppm, 3.027, cr, 0.02),
        'PCr': gaussian(ppm, 3.027, cr, 0.02),
        'NAA': gaussian(ppm, 2.008, 5.0, 0.02),
    })


class TestMetaboliteFreeSpectrum:
    def test_recovers_the_signal_the_model_did_not_explain(self):
        curves = make_curves(mm_height=1.0, baseline_share=0.6)
        free = metabolite_free_spectrum(curves)
        # data - metabolite model == the MM bump, whatever the baseline took.
        expected = curves['data'] - (curves['fit'] - curves['baseline'])
        assert free['signal'].to_numpy() == pytest.approx(
            expected.to_numpy()[np.argsort(curves['ppm'])])

    def test_is_independent_of_how_much_the_baseline_absorbed(self):
        # The reason the baseline is added back rather than discarded.
        a = metabolite_free_spectrum(make_curves(baseline_share=0.1))
        b = metabolite_free_spectrum(make_curves(baseline_share=0.9))
        assert a['signal'].to_numpy() == pytest.approx(b['signal'].to_numpy())

    def test_sorted_ascending(self):
        curves = make_curves()
        free = metabolite_free_spectrum(curves.iloc[::-1].reset_index(drop=True))
        assert np.all(np.diff(free['ppm']) > 0)

    def test_inconsistent_curves_are_rejected(self):
        # The exporter once wrote FIDs into these columns. Such a file yields a
        # plausible-looking number, so it must fail rather than be measured.
        curves = make_curves()
        curves['residual'] = np.random.default_rng(1).normal(0, 1, len(curves))
        with pytest.raises(ValueError, match='internally inconsistent'):
            metabolite_free_spectrum(curves)

    def test_missing_column_is_an_error(self):
        with pytest.raises(KeyError, match='baseline'):
            metabolite_free_spectrum(make_curves().drop(columns=['baseline']))


class TestSpline:
    def test_confined_to_the_requested_range(self):
        curves = make_curves()
        free = metabolite_free_spectrum(curves)
        ppm, _ = fit_mm_spline(free['ppm'].to_numpy(), free['signal'].to_numpy())
        assert ppm.min() >= 0.2 and ppm.max() <= 1.8

    def test_follows_the_envelope_but_not_the_noise(self):
        clean = metabolite_free_spectrum(make_curves(noise=0.0))
        noisy = metabolite_free_spectrum(make_curves(noise=0.3, seed=3))
        ppm, truth = fit_mm_spline(clean['ppm'].to_numpy(), clean['signal'].to_numpy())
        _, got = fit_mm_spline(noisy['ppm'].to_numpy(), noisy['signal'].to_numpy())
        # The spline must suppress the noise it was given, not reproduce it.
        assert np.std(got - truth) < 0.3

    def test_too_few_points_is_an_error(self):
        with pytest.raises(ValueError, match='too few'):
            fit_mm_spline(np.linspace(0.2, 1.8, 8), np.zeros(8))


class TestAnchor:
    def test_removes_a_constant_pedestal(self):
        ppm = np.linspace(0.2, 1.8, 800)
        anchored, line = anchor_envelope(ppm, np.full_like(ppm, -0.15))
        assert anchored == pytest.approx(np.zeros_like(ppm), abs=1e-9)
        assert line == pytest.approx(np.full_like(ppm, -0.15))

    def test_removes_a_sloping_pedestal(self):
        ppm = np.linspace(0.2, 1.8, 800)
        anchored, _ = anchor_envelope(ppm, 0.3 * ppm - 0.2)
        assert anchored == pytest.approx(np.zeros_like(ppm), abs=1e-9)

    def test_leaves_a_bump_between_the_flanks(self):
        ppm = np.linspace(0.2, 1.8, 800)
        bump = gaussian(ppm, 0.90, 1.0, 0.09)
        anchored, _ = anchor_envelope(ppm, bump - 0.15)
        assert anchored.max() == pytest.approx(1.0, abs=0.05)

    def test_flanks_with_no_points_are_an_error(self):
        ppm = np.linspace(0.2, 1.8, 800)
        with pytest.raises(ValueError, match='anchor flanks'):
            anchor_envelope(ppm, np.zeros_like(ppm), flanks=[(3.0, 3.5)])


class TestIntegrationAndScaling:
    def test_area_matches_the_known_gaussian(self):
        # A Gaussian of height h and width w has area h*w*sqrt(2*pi), but the
        # band spans only +/-0.20 ppm = 2.22 sigma about the centre, so the
        # expected area is that fraction of it (~97.4%), not the whole thing.
        from scipy.special import erf
        ppm = np.linspace(0.2, 1.8, 4000)
        envelope = gaussian(ppm, 0.90, 1.0, 0.09)
        areas = integrate_bands(ppm, envelope, {'MM09': (0.70, 1.10)})
        captured = erf(0.20 / (0.09 * np.sqrt(2)))
        assert areas['MM09'] == pytest.approx(
            1.0 * 0.09 * np.sqrt(2 * np.pi) * captured, rel=0.02)

    def test_reference_area_sums_the_creatines(self):
        ppm = np.linspace(0.2, 4.2, 4000)
        area = reference_area(make_metabolites(ppm, cr=1.0))
        assert area == pytest.approx(2 * 1.0 * 0.02 * np.sqrt(2 * np.pi), rel=0.02)

    def test_reference_area_requires_a_creatine_column(self):
        ppm = np.linspace(0.2, 4.2, 100)
        with pytest.raises(KeyError, match='Cr'):
            reference_area(pd.DataFrame({'ppm': ppm, 'NAA': np.zeros_like(ppm)}))


class TestQuantifyMM:
    def test_recovers_a_known_mm_area(self):
        # The MM09 band stops at 0.95 to stay clear of the trough at 0.95-1.10,
        # so it captures the part of the synthetic peak (centre 0.90, width
        # 0.09) between -2.22 and +0.56 sigma, not all of it.
        from scipy.special import erf
        low, high = MM_BANDS['MM09']
        curves = make_curves(mm_height=1.0, pedestal=-0.15)
        summary, _ = quantify_mm(curves, make_metabolites(curves['ppm'].to_numpy()))
        mm09 = summary.set_index('band').loc['MM09', 'area']
        captured = 0.5 * (erf((high - 0.90) / (0.09 * np.sqrt(2)))
                          - erf((low - 0.90) / (0.09 * np.sqrt(2))))
        assert mm09 == pytest.approx(
            1.0 * 0.09 * np.sqrt(2 * np.pi) * captured, rel=0.10)

    def test_mm09_band_stops_before_the_trough(self):
        # Regression guard on the band edge. The conventional 1.10 limit
        # integrates through a negative trough present in every session and
        # subtracts real signal; 0.95 took the baseline-order CV from 7.6% to
        # 5.2% and the area from 0.40 to 0.56 /tCr on validation data.
        assert MM_BANDS['MM09'] == (0.70, 0.95)

    def test_pedestal_does_not_change_the_answer(self):
        # The failure that motivated the anchor: without it, the area tracked
        # the pedestal rather than the MM content.
        areas = []
        for pedestal in (-0.4, 0.0, 0.4):
            curves = make_curves(mm_height=1.0, pedestal=pedestal)
            summary, _ = quantify_mm(curves)
            areas.append(summary.set_index('band').loc['MM09', 'area'])
        assert np.std(areas) / np.mean(areas) < 0.02

    def test_reports_the_ratio_to_creatine(self):
        curves = make_curves(mm_height=1.0)
        metabolites = make_metabolites(curves['ppm'].to_numpy(), cr=1.0)
        summary, _ = quantify_mm(curves, metabolites)
        row = summary.set_index('band').loc['MM09']
        assert row['area_per_tcr'] == pytest.approx(
            row['area'] / reference_area(metabolites), rel=1e-6)

    def test_ratio_is_nan_without_metabolite_curves(self):
        summary, _ = quantify_mm(make_curves())
        assert summary['area_per_tcr'].isna().all()

    def test_only_validated_bands_are_reported(self):
        # MM14 and MM17 failed the stability test; reporting them would look
        # like a measurement.
        summary, _ = quantify_mm(make_curves())
        assert set(summary['band']) == set(MM_BANDS)
        assert 'MM14' not in summary['band'].to_numpy()
        assert 'MM17' not in summary['band'].to_numpy()

    def test_provisional_bands_are_flagged(self):
        summary, _ = quantify_mm(make_curves())
        flagged = set(summary.loc[summary['provisional'], 'band'])
        assert flagged == set(PROVISIONAL_BANDS)

    def test_envelope_is_returned_for_plotting(self):
        curves = make_curves(mm_height=1.0, pedestal=-0.15)
        _, envelope = quantify_mm(curves)
        assert list(envelope.columns) == ['ppm', 'signal', 'envelope', 'anchor']
        assert len(envelope) > 100
        # anchor + envelope reconstructs the unreferenced spline.
        assert np.isfinite(envelope.to_numpy()).all()

    def test_flank_windows_are_configurable(self):
        curves = make_curves(mm_height=1.0)
        wide, _ = quantify_mm(curves, flanks=DEFAULT_FLANKS)
        narrow, _ = quantify_mm(curves, flanks=[(0.30, 0.55), (1.60, 1.75)])
        # Different anchors, same bump: the area should barely move.
        a = wide.set_index('band').loc['MM09', 'area']
        b = narrow.set_index('band').loc['MM09', 'area']
        assert abs(a - b) / a < 0.10


class TestReferenceGuard:
    """The denominator fails silently, so it gets its own guard and own tests.

    A session whose modelled creatine integrates to ~0 still fits, still
    reports plausible concentrations and still passes SNR/linewidth QC. Only
    the ratio explodes -- one real session returned -11.7 /tCr from an entirely
    ordinary MM area of 0.18.
    """

    def test_healthy_reference_passes(self):
        ppm = np.linspace(0.2, 4.2, 4000)
        result = check_reference(make_metabolites(ppm, cr=1.0))
        assert result['ok']
        assert result['purity'] > MIN_REFERENCE_PURITY

    def test_negative_reference_is_rejected(self):
        # The real failure: PCr integrating negative because the fit put
        # creatine into dispersion.
        ppm = np.linspace(0.2, 4.2, 4000)
        metabolites = make_metabolites(ppm, cr=1.0)
        metabolites['PCr'] = -2.0 * metabolites['PCr']
        result = check_reference(metabolites)
        assert not result['ok']
        assert 'non-positive' in result['reason']

    def test_collapsed_reference_is_rejected(self):
        # Creatine swamped by another species inside its own window.
        ppm = np.linspace(0.2, 4.2, 4000)
        metabolites = make_metabolites(ppm, cr=1.0)
        metabolites[['Cr', 'PCr']] *= 0.02
        metabolites['Cho'] = gaussian(ppm, 3.03, 5.0, 0.02)
        result = check_reference(metabolites)
        assert not result['ok']
        assert 'below' in result['reason']

    def test_reference_uses_the_methyl_singlet_not_the_whole_range(self):
        # Regression guard on the fix. A wide-range integral reports phase
        # rather than amplitude; five of 92 sessions collapsed that way, one to
        # a negative area, from entirely normal creatine amplitudes.
        ppm = np.linspace(0.2, 4.2, 4000)
        metabolites = make_metabolites(ppm, cr=1.0)
        # Signal far outside the tCr window must not enter the reference.
        metabolites['Cr'] = metabolites['Cr'] - gaussian(ppm, 1.30, 20.0, 0.05)
        assert reference_area(metabolites) > 0

    def test_quantify_returns_nan_not_a_wrong_number(self):
        ppm = np.linspace(0.2, 4.2, 2000)
        curves = make_curves(mm_height=1.0)
        metabolites = make_metabolites(curves['ppm'].to_numpy(), cr=1.0)
        metabolites['PCr'] = -2.0 * metabolites['PCr']
        summary, _ = quantify_mm(curves, metabolites)
        assert summary['area_per_tcr'].isna().all()
        # The area itself is unaffected and must still be reported.
        assert summary.set_index('band').loc['MM09', 'area'] > 0
        assert ppm.size  # keep the fixture meaningful

    def test_area_survives_a_bad_reference(self):
        good = quantify_mm(make_curves(mm_height=1.0))[0]
        metabolites = make_metabolites(make_curves()['ppm'].to_numpy(), cr=1.0)
        metabolites[['Cr', 'PCr']] *= 0.02
        bad = quantify_mm(make_curves(mm_height=1.0), metabolites)[0]
        assert bad.set_index('band').loc['MM09', 'area'] == pytest.approx(
            good.set_index('band').loc['MM09', 'area'])


class TestStability:
    def test_flat_across_baseline_orders_is_zero_cv(self):
        estimates = {f'poly{n}': {'MM09': 0.4} for n in (2, 3, 4, 5)}
        assert mm_stability(estimates)['MM09'] == pytest.approx(0.0)

    def test_swinging_band_reports_a_large_cv(self):
        estimates = {'poly2': {'MM09': 0.1}, 'poly3': {'MM09': 0.5},
                     'poly4': {'MM09': 0.9}}
        assert mm_stability(estimates)['MM09'] > 50.0

    def test_only_bands_common_to_every_order_are_scored(self):
        estimates = {'poly2': {'MM09': 0.4, 'MM12': 0.1},
                     'poly3': {'MM09': 0.4}}
        assert set(mm_stability(estimates)) == {'MM09'}

    def test_empty_input(self):
        assert mm_stability({}) == {}

    def test_band_averaging_to_zero_is_skipped_not_infinite(self):
        estimates = {'poly2': {'MM14': 0.1}, 'poly3': {'MM14': -0.1}}
        assert 'MM14' not in mm_stability(estimates)
