"""init_study next-steps must emit the current CLIs, not the old signatures."""
from pathlib import Path

from neurofaune.study_initialization import _generate_next_steps


def _report(anat=0, dwi=0, func=0, bruker_scans=0, bids_subjects=0):
    steps = {}
    if bruker_scans:
        steps["bruker_discovery"] = {"status": "success", "n_scans": bruker_scans}
    if bids_subjects:
        steps["bids_discovery"] = {
            "status": "success",
            "n_subjects": bids_subjects,
            "modalities": {"anat": anat, "dwi": dwi, "func": func},
        }
    return {"steps": steps}


def test_bruker_only_points_at_neurofaune_bids():
    study = Path("/study")
    steps = _generate_next_steps(_report(bruker_scans=12), study)
    joined = "\n".join(steps)
    assert "neurofaune bids --config /study/config.yaml" in joined
    assert "brkraw" not in joined


def test_bids_anat_uses_positional_dirs():
    study = Path("/study")
    steps = _generate_next_steps(_report(anat=4, bids_subjects=4), study)
    joined = "\n".join(steps)
    assert "batch_preprocess_anat.py /study/raw/bids /study --config /study/config.yaml" in joined
    assert "build_templates.py --config /study/config.yaml --cohort all --modality anat" in joined


def test_bids_dwi_and_func_use_flags():
    study = Path("/study")
    steps = _generate_next_steps(
        _report(dwi=3, func=2, bids_subjects=3), study,
    )
    joined = "\n".join(steps)
    assert (
        "batch_preprocess_dwi.py --config /study/config.yaml "
        "--bids-root /study/raw/bids --output-root /study"
    ) in joined
    assert (
        "batch_preprocess_func.py --bids-root /study/raw/bids "
        "--output-root /study --config /study/config.yaml"
    ) in joined
