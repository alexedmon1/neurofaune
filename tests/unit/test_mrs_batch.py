"""The MRS cohort batch is part of the installed package (neurofaune mrs), so a study
pinning neurofaune pins it; scripts/batch_preprocess_mrs.py is a wrapper around it.
"""
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from neurofaune.preprocess.cli import main as cli
from neurofaune.preprocess.workflows import mrs_batch
from neurofaune.preprocess.workflows.mrs_batch import (
    find_anat_image,
    find_tissue_maps,
    parse_session_name,
    run_batch,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name, expected", [
    ("IRC1200_Cuprizone_CageCPZ1_Rat1Y_2__Cage_CPZ1__Rat_1Y_1_2_20260408_140507",
     {"subject": "sub-1Y", "session": "ses-2"}),
    ("IRC1200_Cuprizone_CageCPZ2_Rat4y_3__Cage_CPZ2__Rat_4Y_1_3_20260505_090426",
     {"subject": "sub-4Y", "session": "ses-3"}),
    ("IRC1200_Cuprizone_CageCPZ1_Rat1Z_1a__CPZ_scan_20260406_1_2_20260406_104501",
     {"subject": "sub-1Z", "session": "ses-1a"}),
    ("phantom_20260101", None),
])
def test_parse_session_name(name, expected):
    assert parse_session_name(name) == expected


def test_tissue_maps_need_all_three(tmp_path):
    anat = tmp_path / "sub-1Y" / "ses-1" / "anat"
    anat.mkdir(parents=True)
    for label in ("GM", "WM"):
        (anat / f"sub-1Y_ses-1_label-{label}_probseg.nii.gz").write_text("")
    assert find_tissue_maps(tmp_path, "sub-1Y", "ses-1") is None
    (anat / "sub-1Y_ses-1_label-CSF_probseg.nii.gz").write_text("")
    assert set(find_tissue_maps(tmp_path, "sub-1Y", "ses-1")) == {"GM", "WM", "CSF"}


def test_anat_image_prefers_the_preprocessed_t2w(tmp_path):
    anat = tmp_path / "sub-1Y" / "ses-1" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-1Y_ses-1_run-1_T2w.nii.gz").write_text("")
    assert find_anat_image(tmp_path, "sub-1Y", "ses-1").name == "sub-1Y_ses-1_run-1_T2w.nii.gz"
    (anat / "sub-1Y_ses-1_desc-preproc_T2w.nii.gz").write_text("")
    assert find_anat_image(tmp_path, "sub-1Y", "ses-1").name == "sub-1Y_ses-1_desc-preproc_T2w.nii.gz"


def test_run_batch_refuses_a_missing_bruker_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_batch(tmp_path / "absent", tmp_path / "mrs", tmp_path / "config.yaml", dry_run=True)


def test_dry_run_writes_nothing(tmp_path, capsys):
    bruker = tmp_path / "bruker"
    bruker.mkdir()
    out = tmp_path / "mrs"
    assert run_batch(bruker, out, tmp_path / "config.yaml", dry_run=True) == 0
    assert "Found 0 sessions" in capsys.readouterr().out
    assert not out.exists()


def test_dry_run_lists_discovered_sessions(tmp_path, monkeypatch, capsys):
    bruker = tmp_path / "bruker"
    (bruker / "X_Rat2C_1__Y").mkdir(parents=True)
    monkeypatch.setattr(mrs_batch, "find_press_scans", lambda d: [
        {"water_suppressed": False, "n_averages": 16, "scan_number": 12, "protocol": "ref",
         "echo_time": 20.0, "voxel_size": [7.5, 2, 2]},
        {"water_suppressed": True, "n_averages": 256, "scan_number": 13, "protocol": "PRESS",
         "echo_time": 20.0, "voxel_size": [7.5, 2, 2]},
    ])
    monkeypatch.setattr(mrs_batch, "find_anat_scan", lambda d: "5")
    assert run_batch(bruker, tmp_path / "mrs", tmp_path / "config.yaml",
                     derivatives=tmp_path / "derivatives", dry_run=True) == 0
    out = capsys.readouterr().out
    assert "Found 1 sessions" in out and "sub-2C" in out and "0/1 sessions have a T2w segmentation" in out


def test_cli_mrs_dry_run_and_errors(tmp_path):
    bruker = tmp_path / "bruker"
    bruker.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("study: {name: t}\n")
    runner = CliRunner()
    ok = runner.invoke(cli, ["mrs", str(bruker), str(tmp_path / "mrs"), "--config", str(config), "--dry-run"])
    assert ok.exit_code == 0, ok.output
    assert "Found 0 sessions" in ok.output
    missing = runner.invoke(cli, ["mrs", str(tmp_path / "absent"), str(tmp_path / "mrs"),
                                  "--config", str(config), "--dry-run"])
    assert missing.exit_code == 2


def test_script_wrapper_runs_the_package_main():
    script = REPO / "scripts" / "batch_preprocess_mrs.py"
    assert "from neurofaune.preprocess.workflows.mrs_batch import main" in script.read_text()
    res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert res.returncode == 0 and "bruker_root" in res.stdout
