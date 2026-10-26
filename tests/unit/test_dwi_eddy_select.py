"""Which eddy build runs. CPU and GPU eddy outputs differ (FAC600 mouse: median
|dFA| 4e-3), so the choice is a study setting and must not drift with the machine.
"""
import pytest

from neurofaune.preprocess.workflows.dwi_preprocess import find_cuda_build, is_cuda_eddy, select_eddy


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
    assert select_eddy({}, True, which=_which({"eddy", "eddy_cpu"}),
                       find_cuda=lambda name: None) == "eddy_cpu"
    assert "do not mix" in capsys.readouterr().out


VERSIONED = "/usr/local/fsl/bin/eddy_cuda11.0"


def test_versioned_cuda_build_is_used_when_plain_name_is_absent(capsys):
    """FSL >= 6.0.7 installs eddy_cuda11.0 (etc.) and no plain eddy_cuda; its
    find_cuda_exe resolves the build for the driver, as its own dispatcher does."""
    asked = []

    def find_cuda(name):
        asked.append(name)
        return VERSIONED

    got = select_eddy({}, None, which=_which({"eddy", "eddy_cpu"}), find_cuda=find_cuda)
    assert got == VERSIONED
    assert asked == ["eddy_cuda"]
    assert "WARNING" not in capsys.readouterr().out


def test_plain_eddy_cuda_on_path_wins_over_the_versioned_lookup():
    def find_cuda(name):
        raise AssertionError("find_cuda_exe should not be consulted")

    assert select_eddy({}, True, which=_which(ALL), find_cuda=find_cuda) == "eddy_cuda"


def test_cpu_request_never_looks_for_a_cuda_build():
    def find_cuda(name):
        raise AssertionError("find_cuda_exe should not be consulted for CPU eddy")

    assert select_eddy({}, False, which=_which({"eddy", "eddy_cpu"}), find_cuda=find_cuda) == "eddy_cpu"


@pytest.mark.parametrize("cmd, cuda", [
    ("eddy_cuda", True),
    (VERSIONED, True),
    ("eddy_cuda10.2", True),
    ("eddy_cpu", False),
    ("eddy", False),
    ("/usr/local/fsl/bin/eddy_cpu", False),
])
def test_is_cuda_eddy_recognises_versioned_builds(cmd, cuda):
    """The eddy call adds --very_verbose for GPU builds and --nthr for CPU ones."""
    assert is_cuda_eddy(cmd) is cuda


def test_find_cuda_build_returns_none_without_fsl(monkeypatch):
    monkeypatch.delenv("FSLDIR", raising=False)
    assert find_cuda_build("eddy_cuda", which=lambda name: None) is None


def test_find_cuda_build_reads_find_cuda_exe(monkeypatch, tmp_path):
    """A stand-in find_cuda_exe that prints a build path, as FSL's does, and prints
    nothing for a build it does not have (FSL's behaviour), which gives None."""
    build = tmp_path / "eddy_cuda11.0"
    build.write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tool = bindir / "find_cuda_exe"
    tool.write_text(f'#!/bin/sh\n[ "$1" = eddy_cuda ] && echo {build}\nexit 0\n')
    tool.chmod(0o755)
    monkeypatch.setenv("FSLDIR", str(tmp_path))
    assert find_cuda_build("eddy_cuda", which=lambda name: None) == str(build)
    assert find_cuda_build("eddy_other", which=lambda name: None) is None
