"""TBSS accepts every diffusion measure the DWI workflow produces, not just DTI."""
import subprocess
import sys
from pathlib import Path

import pytest

from neurofaune.analysis.tbss import prepare_tbss as pt
from neurofaune.templates.sigma_warp import DWI_SIGMA_METRICS


def test_diffusion_metrics_follow_the_canonical_list():
    assert pt.DIFFUSION_METRICS == list(DWI_SIGMA_METRICS)
    for family in (["FA", "MD", "AD", "RD"], ["MK", "AK", "RK", "KFA"], ["FICVF", "ODI", "FISO"]):
        assert set(family) <= set(pt.DIFFUSION_METRICS)


def test_default_stays_tensor_only():
    assert pt.DTI_METRICS == ["FA", "MD", "AD", "RD"]


@pytest.mark.parametrize("metric,name", [
    ("FA", "sub-1X_ses-1_FA.nii.gz"),
    ("MK", "sub-1X_ses-1_model-DKI_MK.nii.gz"),
    ("FICVF", "sub-1X_ses-1_model-NODDI_FICVF.nii.gz"),
])
def test_native_finder_knows_model_entities(tmp_path, metric, name):
    (tmp_path / name).touch()
    assert pt._find_metric_file(tmp_path, "sub-1X", "ses-1", metric) == tmp_path / name


def test_cli_accepts_kurtosis_and_noddi(tmp_path):
    """--metrics used to reject anything but FA/MD/AD/RD at argument parsing."""
    cfg = tmp_path / "missing.yaml"
    res = subprocess.run(
        [sys.executable, "-m", "neurofaune.analysis.tbss.prepare_tbss",
         "--config", str(cfg), "--output-dir", str(tmp_path / "out"),
         "--metrics", "FA", "MK", "RK", "FICVF", "ODI", "--dry-run"],
        capture_output=True, text=True)
    assert "invalid choice" not in res.stderr


def test_cli_still_rejects_unknown_metrics(tmp_path):
    res = subprocess.run(
        [sys.executable, "-m", "neurofaune.analysis.tbss.prepare_tbss",
         "--config", str(tmp_path / "c.yaml"), "--output-dir", str(tmp_path / "o"),
         "--metrics", "MWF"], capture_output=True, text=True)
    assert "invalid choice" in res.stderr
