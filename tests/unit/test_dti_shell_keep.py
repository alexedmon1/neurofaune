"""max_bval selects shells by nominal b, not raw effective b (Bruker writes b + a few %)."""
import numpy as np

from neurofaune.preprocess.workflows.dwi_preprocess import dti_shell_keep

# FAC600 pilot effective b-values: b0 33.8, shells 507-527, 1005-1031, 2006-2042
FAC600 = np.array([33.8] * 5 + list(np.linspace(507, 527, 20)) + list(np.linspace(1005, 1031, 20))
                  + list(np.linspace(2006, 2042, 30)))


def test_b1000_shell_kept_whole_despite_effective_b_above_1000():
    keep = dti_shell_keep(FAC600, 1000)
    assert keep.sum() == 45 and keep[:45].all() and not keep[45:].any()


def test_raw_threshold_would_have_dropped_it():
    assert (FAC600 <= 1000).sum() == 25


def test_b0_always_kept_and_all_kept_when_threshold_high():
    assert dti_shell_keep(FAC600, 500).sum() == 25
    assert dti_shell_keep(FAC600, 2000).all()
