"""Per-run functional output names (functional.output_entities)."""
from pathlib import Path

from neurofaune.preprocess.workflows.func_preprocess import func_output_entities
from neurofaune.provenance import write_provenance

ON = {"functional": {"output_entities": True}}
AX = Path("sub-MOUSE1_ses-1_task-rest_acq-ax_run-21_bold.nii.gz")


def test_off_by_default_keeps_legacy_names():
    assert func_output_entities(AX, {}) == ""


def test_keeps_task_acq_run_in_bids_order():
    assert func_output_entities(AX, ON) == "task-rest_acq-ax_run-21"


def test_two_runs_of_a_session_get_distinct_names():
    cor = Path("sub-MOUSE1_ses-1_task-rest_acq-cor_run-23_bold.nii.gz")
    assert func_output_entities(AX, ON) != func_output_entities(cor, ON)


def test_echo_dropped_for_multiecho_lists():
    echoes = [Path("sub-1_ses-1_task-rest_run-2_echo-1_bold.nii.gz"),
              Path("sub-1_ses-1_task-rest_run-2_echo-2_bold.nii.gz")]
    assert func_output_entities(echoes, ON) == "task-rest_run-2"


def test_no_entities_in_name_gives_empty():
    assert func_output_entities(Path("sub-Rat1_ses-p60_bold.nii.gz"), ON) == ""


def test_provenance_file_is_per_run(tmp_path):
    a = write_provenance(tmp_path, "sub-1", "ses-1", "func", entities="task-rest_acq-ax_run-21")
    b = write_provenance(tmp_path, "sub-1", "ses-1", "func", entities="task-rest_acq-cor_run-23")
    assert a != b and a.name == "sub-1_ses-1_task-rest_acq-ax_run-21_func-provenance.json"
    assert write_provenance(tmp_path, "sub-1", "ses-1", "func").name == "sub-1_ses-1_func-provenance.json"
