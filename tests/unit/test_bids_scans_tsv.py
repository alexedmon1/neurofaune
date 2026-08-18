"""BIDS must record WHICH raw session it converted.

The converter writes images with a scaled-identity affine, so a BIDS NIfTI
carries no scanner position. Anything needing the real geometry -- spectroscopy
reads both its voxel (PVM_VoxelGeoCub) and the anatomical slice package
(PVM_SPackArrGradOrient/SliceOffset/Fov) from the raw method files -- has to go
back to the raw session directory.

Recomputing that directory from the session label is not safe, because session
relabelling is many-to-one. That is not hypothetical: on the cuprizone study
'Rat1Z_1a' is relabelled to ses-1 while the aborted 'Rat1Z_1' still sits beside
it on disk, so a label-based lookup silently reads a real FID from the wrong
acquisition and produces plausible numbers for the wrong scan.
"""
import re

from neurofaune.utils.bids import SCANS_TSV_COLS, parse_session_name

RGX = re.compile(
    r"^IRC\d+_\w+_CageCPZ(?P<cage>\w+?)_Rat(?P<subject>\d+[A-Za-z])_(?P<session>\d+[a-z]?)__"
)
RELABEL = {"1a": "1"}          # the cuprizone mapping


def test_scans_tsv_records_the_raw_session_directory():
    assert "source_session_dir" in SCANS_TSV_COLS


def test_session_relabel_is_many_to_one_so_the_label_cannot_be_inverted():
    """Two distinct raw sessions collapse onto one label -- one of them aborted."""
    aborted = parse_session_name(
        "IRC1200_Cuprizone_CageCPZ1_Rat1Z_1__CPZ_scan_20260406_1_1_20260406_103911",
        RGX, RELABEL)
    real = parse_session_name(
        "IRC1200_Cuprizone_CageCPZ1_Rat1Z_1a__CPZ_scan_20260406_1_2_20260406_104501",
        RGX, RELABEL)

    assert aborted["subject"] == real["subject"] == "1Z"
    # both land on ses-1: the label alone cannot say which directory to read
    assert aborted["session"] == real["session"] == "1"
