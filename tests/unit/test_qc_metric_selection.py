"""Which metrics a cohort-relative z-test may be applied to, and on which tail.

Every case here is a defect observed on a real 92-session rodent cohort, where
~30 of 41 QC flags turned out to indict the metric rather than the session.
"""
import numpy as np
import pandas as pd
import pytest

from neurofaune.preprocess.qc.batch_summary import (
    BatchQCConfig,
    detect_outliers,
    select_zscore_metrics,
    _metric_direction,
)


def _cohort(n=92, **cols):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'subject': [f'sub-{i:03d}' for i in range(n)],
        'session': ['ses-1'] * n,
        'plain_metric': rng.normal(10, 1, n),
    })
    for k, v in cols.items():
        df[k] = v
    return df


class TestDegenerateMetrics:
    def test_float_noise_variance_is_not_z_scored(self):
        """csf_max had 2 distinct values, sd 4e-17 -> z=5.5 on zero variance."""
        vals = np.full(92, 1.0)
        vals[:2] = 1.0 + 1e-16
        df = _cohort(csf_ceiling=vals)
        usable, skipped = select_zscore_metrics(df)
        assert 'csf_ceiling' in skipped
        assert 'csf_ceiling' not in usable

    def test_saturated_metric_is_not_z_scored(self):
        """A metric pinned at a physical ceiling ranks noise, not quality."""
        vals = np.full(92, 1.0)
        vals[:10] = np.linspace(0.90, 0.99, 10)   # 89% saturated at 1.0
        df = _cohort(fraction_metric=vals)
        _, skipped = select_zscore_metrics(df)
        assert 'fraction_metric' in skipped
        assert 'saturated' in skipped['fraction_metric']

    def test_too_few_distinct_values(self):
        df = _cohort(three_valued=np.tile([1.0, 2.0, 3.0, 1.0], 23))
        _, skipped = select_zscore_metrics(df)
        assert 'distinct values' in skipped['three_valued']

    def test_boolean_constant_column_is_skipped(self):
        """anat potential_over_stripping was True for 92/92 sessions."""
        df = _cohort(potential_over_stripping=np.ones(92))
        _, skipped = select_zscore_metrics(df)
        assert 'potential_over_stripping' in skipped

    def test_healthy_metric_survives(self):
        usable, skipped = select_zscore_metrics(_cohort())
        assert 'plain_metric' in usable
        assert 'plain_metric' not in skipped


class TestOrderStatistics:
    @pytest.mark.parametrize('name', ['fa_max', 'md_min', 'mwf_max', 'iwf_max'])
    def test_per_map_extremes_are_excluded(self, name):
        """fa_max exceeded 1 in 100% of sessions; the flagged ones were least extreme."""
        rng = np.random.default_rng(1)
        df = _cohort(**{name: rng.normal(1.2, 0.05, 92)})
        usable, skipped = select_zscore_metrics(df)
        assert name in skipped
        assert name not in usable


class TestDirection:
    @pytest.mark.parametrize('name', [
        'skull_stripping_snr_estimate', 'registration_nmi', 'to_sigma_dice',
        'min_snr', 'to_template_correlation', 'cb_covered',
    ])
    def test_goodness_metrics_flag_low_only(self, name):
        assert _metric_direction(name) == 'low'

    @pytest.mark.parametrize('name', [
        'motion_mean_fd', 'motion_max_fd', 'mean_dvars_std',
        'pct_bad_volumes', 'to_sigma_centroid_offset_mm',
    ])
    def test_badness_metrics_flag_high_only(self, name):
        assert _metric_direction(name) == 'high'

    def test_unknown_metric_stays_two_sided(self):
        assert _metric_direction('t2_median') == 'both'

    def test_best_session_on_a_goodness_metric_is_not_flagged(self):
        """The observed defect: a two-sided test flags the BEST sessions."""
        rng = np.random.default_rng(2)
        snr = rng.normal(10, 1, 92)
        snr[0] = 20.0          # by far the best session in the cohort
        df = _cohort(registration_nmi=snr)
        out = detect_outliers(df, BatchQCConfig(), 'func')
        assert 'registration_nmi' not in out.iloc[0]['flags']

    def test_worst_session_on_a_goodness_metric_is_still_flagged(self):
        rng = np.random.default_rng(3)
        snr = rng.normal(10, 1, 92)
        snr[0] = 1.0           # genuinely bad
        df = _cohort(registration_nmi=snr)
        out = detect_outliers(df, BatchQCConfig(), 'func')
        assert 'registration_nmi' in out.iloc[0]['flags']


class TestScaleDependentMetrics:
    def test_raw_dvars_is_skipped_but_its_standardized_twin_is_not(self):
        rng = np.random.default_rng(4)
        df = _cohort(motion_mean_dvars=rng.normal(34, 7, 92),
                     motion_mean_dvars_std=rng.normal(1.0, 0.1, 92))
        usable, skipped = select_zscore_metrics(df)
        assert 'motion_mean_dvars' in skipped
        assert 'motion_mean_dvars_std' in usable


class TestAbsoluteGates:
    def test_gate_tripping_the_whole_cohort_is_a_config_defect(self):
        """dwi fa_mean (0.25, 0.55) tripped 90/92 on a cohort running at 0.202."""
        rng = np.random.default_rng(5)
        df = _cohort(fa_mean=rng.normal(0.202, 0.01, 92))
        cfg = BatchQCConfig()
        cfg.dwi_thresholds = {'fa_mean': (0.25, 0.55)}
        out = detect_outliers(df, cfg, 'dwi')
        assert 'fa_mean' in out.attrs['threshold_misconfigured']
        # No session carries the ABSOLUTE flag ("fa_mean=..."); a z-flag on the
        # same metric is a different statement and stays allowed.
        assert not out['flags'].str.contains('fa_mean=').any()

    def test_a_discriminating_gate_still_flags_sessions(self):
        rng = np.random.default_rng(6)
        vals = rng.normal(0.40, 0.02, 92)
        vals[:3] = 0.10                       # 3 of 92 genuinely below
        df = _cohort(fa_mean=vals)
        cfg = BatchQCConfig()
        cfg.dwi_thresholds = {'fa_mean': (0.25, 0.55)}
        out = detect_outliers(df, cfg, 'dwi')
        assert 'fa_mean' not in out.attrs['threshold_misconfigured']
        assert out['flags'].str.contains('fa_mean=').sum() == 3

    def test_boundary_below_half_still_gates(self):
        vals = np.concatenate([np.full(45, 0.1), np.full(47, 0.4)])
        df = _cohort(fa_mean=vals)
        cfg = BatchQCConfig()
        cfg.dwi_thresholds = {'fa_mean': (0.25, None)}
        out = detect_outliers(df, cfg, 'dwi')
        assert 'fa_mean' not in out.attrs['threshold_misconfigured']
        assert out['flags'].str.contains('fa_mean=').sum() == 45

    def test_misconfigured_report_names_the_observed_median(self):
        df = _cohort(fa_mean=np.full(92, 0.202))
        cfg = BatchQCConfig()
        cfg.dwi_thresholds = {'fa_mean': (0.25, 0.55)}
        out = detect_outliers(df, cfg, 'dwi')
        assert '0.202' in out.attrs['threshold_misconfigured']['fa_mean']


class TestContract:
    def test_absolute_thresholds_still_apply_to_skipped_metrics(self):
        """Excluding a metric from z-scoring must not exempt it from a hard gate.

        fa_max is never z-scored (order statistic), but a study that sets an
        explicit ceiling on it must still get session flags -- as long as the
        gate discriminates, i.e. trips a minority.
        """
        vals = np.full(92, 0.9)
        vals[:4] = 2.0                        # 4 of 92 above the ceiling
        df = _cohort(fa_max=vals)
        cfg = BatchQCConfig()
        cfg.dwi_thresholds = {'fa_max': (None, 1.0)}
        out = detect_outliers(df, cfg, 'dwi')
        assert 'fa_max' in out.attrs['zscore_skipped']
        assert 'fa_max' not in out.attrs['threshold_misconfigured']
        assert out['flags'].str.contains('fa_max=').sum() == 4
        assert 'fa_max' in out.iloc[0]['flags']

    def test_skips_are_reported_not_silent(self):
        """A metric dropped from QC must not look like a metric that passed."""
        df = _cohort(fa_max=np.full(92, 1.2))
        out = detect_outliers(df, BatchQCConfig(), 'dwi')
        assert 'fa_max' in out.attrs['zscore_skipped']

    def test_return_shape_unchanged(self):
        out = detect_outliers(_cohort(), BatchQCConfig(), 'func')
        assert list(out.columns) == ['subject', 'session', 'is_outlier',
                                     'n_flags', 'flags']
        assert len(out) == 92

    def test_empty_input(self):
        assert detect_outliers(pd.DataFrame(), BatchQCConfig(), 'func').empty
