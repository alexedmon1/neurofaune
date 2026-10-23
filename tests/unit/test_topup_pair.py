"""find_topup_pair: the reverse-PE pair that can unwarp a series, from BIDS fmap/ (no IntendedFor)."""
import json

import nibabel as nib
import numpy as np

from neurofaune.preprocess.utils.topup import find_topup_pair


def _img(path, shape=(8, 8, 5), zoom=1.0, pe="j-", trt=0.016, nvol=None):
    aff = np.diag([zoom, zoom, zoom, 1.0])
    data = np.zeros(shape + ((nvol,) if nvol else ()), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, aff), str(path))
    path.with_name(path.name.replace(".nii.gz", ".json")).write_text(
        json.dumps({"PhaseEncodingDirection": pe, "TotalReadoutTime": trt}))
    return path


def _session(tmp_path):
    ses = tmp_path / "sub-1" / "ses-1"
    bold = _img(ses / "func" / "sub-1_ses-1_task-rest_run-21_bold.nii.gz", nvol=4)
    return ses, bold


def test_finds_same_and_reverse_pe_on_the_grid(tmp_path):
    ses, bold = _session(tmp_path)
    fwd = _img(ses / "fmap" / "sub-1_ses-1_acq-ge_dir-fwd_run-14_epi.nii.gz", pe="j-")
    rev = _img(ses / "fmap" / "sub-1_ses-1_acq-ge_dir-rev_run-15_epi.nii.gz", pe="j")
    assert find_topup_pair(bold) == (fwd, rev)


def test_rejects_other_grid_readout_or_axis(tmp_path):
    ses, bold = _session(tmp_path)
    f = ses / "fmap"
    _img(f / "sub-1_ses-1_acq-se_dir-fwd_run-11_epi.nii.gz", shape=(16, 16, 5), pe="j-")   # DWI grid
    _img(f / "sub-1_ses-1_acq-se_dir-rev_run-12_epi.nii.gz", shape=(16, 16, 5), pe="j")
    _img(f / "sub-1_ses-1_acq-x_dir-fwd_run-1_epi.nii.gz", pe="j-", trt=0.025)               # other readout
    _img(f / "sub-1_ses-1_acq-x_dir-rev_run-2_epi.nii.gz", pe="j", trt=0.025)
    _img(f / "sub-1_ses-1_acq-y_dir-rev_run-3_epi.nii.gz", pe="i")                           # other axis
    assert find_topup_pair(bold) is None


def test_needs_both_directions(tmp_path):
    ses, bold = _session(tmp_path)
    _img(ses / "fmap" / "sub-1_ses-1_dir-fwd_run-14_epi.nii.gz", pe="j-")
    assert find_topup_pair(bold) is None


def test_nearest_run_wins(tmp_path):
    ses, bold = _session(tmp_path)
    f = ses / "fmap"
    _img(f / "sub-1_ses-1_dir-fwd_run-2_epi.nii.gz", pe="j-")
    _img(f / "sub-1_ses-1_dir-rev_run-3_epi.nii.gz", pe="j")
    fwd = _img(f / "sub-1_ses-1_dir-fwd_run-20_epi.nii.gz", pe="j-")
    rev = _img(f / "sub-1_ses-1_dir-rev_run-22_epi.nii.gz", pe="j")
    assert find_topup_pair(bold) == (fwd, rev)
