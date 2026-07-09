"""Unit tests for the per-slice bet timeout in adaptive skull stripping.

Pins the guard added after a batch func run wedged for ~15h on FSL bet
processes that hung on degenerate slices: ``test_bet_frac_on_slice`` must bound
the bet subprocess with a timeout and degrade to a failed-frac result
``(None, 0)`` instead of blocking forever.
"""
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neurofaune.preprocess.utils.func import skull_strip_adaptive as ssa


class _HangingProc:
    """Fake Popen whose communicate() times out, standing in for a wedged bet."""
    def __init__(self, *a, **kw):
        self.pid = 424242
        self.kwargs = kw
    def communicate(self, timeout=None):
        self.timeout = timeout
        raise subprocess.TimeoutExpired(cmd='bet', timeout=timeout)
    def wait(self):
        return -9


def test_bet_timeout_kills_group_returns_failure(tmp_path, monkeypatch):
    calls = {}
    proc_holder = {}

    def fake_popen(cmd, **kwargs):
        p = _HangingProc(cmd, **kwargs)
        proc_holder['p'] = p
        return p

    def fake_getpgid(pid):
        calls['getpgid_pid'] = pid
        return pid

    def fake_killpg(pgid, sig):
        calls['killpg'] = (pgid, sig)

    monkeypatch.setattr(ssa.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(ssa.os, 'getpgid', fake_getpgid)
    monkeypatch.setattr(ssa.os, 'killpg', fake_killpg)

    slice_data = np.random.default_rng(0).random((32, 32)) * 100
    mask, n = ssa.test_bet_frac_on_slice(
        slice_data=slice_data, slice_idx=3, work_dir=tmp_path,
        frac=0.4, cog=(16.0, 16.0), bet_timeout=5.0,
    )

    assert mask is None and n == 0                      # failed frac, not a hang
    assert proc_holder['p'].kwargs.get('start_new_session') is True  # own group
    assert proc_holder['p'].timeout == 5.0              # timeout applied
    # the WHOLE group is killed (wrapper + bet2), not just the direct child
    assert calls['killpg'] == (424242, signal.SIGKILL)
    # temp input cleaned up on timeout (no leaked slice files)
    assert not list(tmp_path.glob('*.nii.gz'))


def test_default_timeout_is_bounded():
    import inspect
    sig = inspect.signature(ssa.test_bet_frac_on_slice)
    assert sig.parameters['bet_timeout'].default == ssa.BET_SLICE_TIMEOUT_S
    assert 0 < ssa.BET_SLICE_TIMEOUT_S < 600   # bounded, sane
