"""neurofaune.results re-exports Lumière (lumiere, neurobox-dev until 2026-11-10), where the specification moved on 2026-11-03."""
import subprocess
import sys

import lumiere


def test_every_submodule_is_the_lumiere_module_itself():
    import neurofaune.results as nr
    from neurofaune.results import _schema, check, gallery, spec, write

    for name in nr.SUBMODULES:
        assert sys.modules[f"neurofaune.results.{name}"] is sys.modules[f"lumiere.{name}"]
    assert (spec, write, _schema, gallery) == (lumiere.spec, lumiere.write, lumiere._schema, lumiere.gallery)
    assert check is lumiere.check is nr.check  # `check` is still the function
    assert callable(check) and not isinstance(check, type(sys))
    assert nr.SPEC == "neurofaune.results" and nr.__all__ == lumiere.__all__


def test_old_import_paths_keep_working():
    from neurofaune.results.check import check_analysis
    from neurofaune.results.spec import analysis_id, run_folder
    from neurofaune.results.write import now

    assert check_analysis is sys.modules["lumiere.check"].check_analysis
    assert analysis_id("dwi", "tbss", "demo") == "dwi/tbss/demo"
    assert callable(run_folder) and callable(now)


def test_the_old_checker_command_still_runs(tmp_path):
    out = subprocess.run([sys.executable, "-m", "neurofaune.results", "check", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 1  # an empty folder holds nothing to check, which is not a pass
    assert "no analysis folders" in out.stderr


def _without_lumiere(monkeypatch):
    """Make neurofaune behave as if the optional results package were not installed."""
    import neurofaune.results as nr
    monkeypatch.setattr(nr, "AVAILABLE", False)


def test_without_the_results_package_tables_are_written_and_the_json_skipped(tmp_path, monkeypatch, caplog):
    import pandas as pd

    from neurofaune.analysis.stats import readout_results as rr

    _without_lumiere(monkeypatch)
    tests = pd.DataFrame({"metric": ["FA"], "contrast": ["c1"]})
    result = rr.write_readout_results(
        tmp_path, tests, pd.DataFrame(), analysis_id=rr.analysis_id("dwi", "tbss", "demo"),
        title="t", description="d", analysis_type="tbss", measure_column="metric", measures=["FA"],
        run_dirs={"FA": tmp_path}, n_permutations=10, alpha=0.05, mask_name="mask", space="SIGMA", inference="TFCE")
    assert result is None
    assert (tmp_path / "tests.csv").exists() and (tmp_path / "clusters.csv").exists()
    assert not (tmp_path / "analysis.json").exists() and not (tmp_path / "tests.json").exists()
    assert "neurofaune[results]" in caplog.text
    assert rr.analysis_id("dwi", "tbss", "demo") == "dwi/tbss/demo"
    assert rr.now().endswith("+00:00")


def test_config_checks_need_no_results_package(monkeypatch):
    from neurofaune.atlas.study_space import study_space_axes

    _without_lumiere(monkeypatch)
    assert study_space_axes({"atlas": {"study_space": {"axes": "lia"}}}) == "LIA"


def test_the_check_command_says_how_to_install_it(monkeypatch, tmp_path, capsys):
    import importlib

    import neurofaune.results.__main__ as cli_module

    _without_lumiere(monkeypatch)
    cli_module = importlib.reload(cli_module)
    try:
        assert cli_module.run_check(str(tmp_path)) == 2
        assert "neurofaune[results]" in capsys.readouterr().err
    finally:
        monkeypatch.undo()
        importlib.reload(cli_module)
