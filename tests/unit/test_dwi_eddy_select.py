"""Which eddy build runs. CPU and GPU eddy outputs differ (FAC600 mouse: median
|dFA| 4e-3), so the choice is a study setting and must not drift with the machine.
"""
import pytest

from neurofaune.preprocess.workflows.dwi_preprocess import select_eddy


def _which(available):
    return lambda name: f"/fsl/bin/{name}" if name in available else None


ALL = {"eddy", "eddy_cpu", "eddy_cuda"}


def test_gpu_is_the_default():
    assert select_eddy({}, None, which=_which(ALL)) == "eddy_cuda"


def test_study_config_can_pin_cpu():
    cfg = {"diffusion": {"eddy": {"use_cuda": False}}}
    assert select_eddy(cfg, None, which=_which(ALL)) == "eddy_cpu"


@pytest.mark.parametrize("use_gpu, expected", [(True, "eddy_cuda"), (False, "eddy_cpu")])
def test_explicit_argument_overrides_config(use_gpu, expected):
    cfg = {"diffusion": {"eddy": {"use_cuda": not use_gpu}}}
    assert select_eddy(cfg, use_gpu, which=_which(ALL)) == expected


def test_cpu_never_uses_the_dispatcher_when_eddy_cpu_exists():
    """FSL's 'eddy' execs eddy_cuda whenever a GPU is visible, ignoring the request."""
    assert select_eddy({}, False, which=_which(ALL)) == "eddy_cpu"
    assert select_eddy({}, False, which=_which({"eddy"})) == "eddy"


def test_missing_cuda_warns_and_falls_back_to_cpu(capsys):
    assert select_eddy({}, True, which=_which({"eddy", "eddy_cpu"})) == "eddy_cpu"
    assert "do not mix" in capsys.readouterr().out
