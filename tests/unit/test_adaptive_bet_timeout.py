"""Unit tests for the per-slice bet timeout in adaptive skull stripping.

Pins the guard added after a batch func run wedged for ~15h on FSL bet
processes that hung on degenerate slices: ``test_bet_frac_on_slice`` must bound
the bet subprocess with a timeout and degrade to a failed-frac result
``(None, 0)`` instead of blocking forever.
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neurofaune.preprocess.utils.func import skull_strip_adaptive as ssa


def test_bet_timeout_returns_failure_not_hang(tmp_path, monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        # bet is invoked with a bounded timeout, and here it "hangs"
        calls['timeout'] = kwargs.get('timeout')
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get('timeout'))

    monkeypatch.setattr(ssa.subprocess, 'run', fake_run)

    slice_data = np.random.default_rng(0).random((32, 32)) * 100
    mask, n = ssa.test_bet_frac_on_slice(
        slice_data=slice_data, slice_idx=3, work_dir=tmp_path,
        frac=0.4, cog=(16.0, 16.0), bet_timeout=5.0,
    )

    assert mask is None and n == 0            # failed frac, not a hang
    assert calls['timeout'] == 5.0            # timeout actually passed to bet
    # temp input must be cleaned up on timeout (no leaked slice files)
    assert not list(tmp_path.glob('*.nii.gz'))


def test_default_timeout_is_bounded():
    import inspect
    sig = inspect.signature(ssa.test_bet_frac_on_slice)
    assert sig.parameters['bet_timeout'].default == ssa.BET_SLICE_TIMEOUT_S
    assert 0 < ssa.BET_SLICE_TIMEOUT_S < 600   # bounded, sane
