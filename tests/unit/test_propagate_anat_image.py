"""Propagating anat-space images into a modality's space.

The regression this guards: tissue probability maps were brought into BOLD
space with nibabel's resample_from_to, which applies no registration. When the
two acquisitions do not share a frame it returns an image of the right shape
sampled from the wrong anatomy, and nothing raises.
"""
import subprocess
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.registration_utils import propagate_anat_image


@pytest.fixture
def anat_and_moving(tmp_path):
    """An anat image and a moving ref whose affines disagree by a translation."""
    data = np.zeros((20, 20, 10), dtype=np.float32)
    data[5:15, 5:15, 2:8] = 1.0

    anat = tmp_path / "anat_probseg.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), anat)

    # Moving reference sits 30 mm away -- the independently-planned-FOV case.
    moving_affine = np.eye(4)
    moving_affine[:3, 3] = [30.0, -25.0, 12.0]
    moving = tmp_path / "moving_ref.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 10), np.float32), moving_affine),
             moving)
    return anat, moving


def test_applies_the_transform_inverted(anat_and_moving, tmp_path):
    """anat -> moving is the INVERSE of the stored moving -> anat affine."""
    anat, moving = anat_and_moving
    affine = tmp_path / "moving_to_anat_0GenericAffine.mat"
    affine.write_bytes(b"")
    out = tmp_path / "out.nii.gz"

    with mock.patch.object(subprocess, "run") as run:
        propagate_anat_image(
            anat_image=anat, moving_ref=moving,
            moving_to_anat_affine=affine, out_image=out,
        )

    cmd = run.call_args[0][0]
    assert cmd[0] == "antsApplyTransforms"
    # The ANTs '[transform,1]' suffix is what inverts it.
    assert f"[{affine},1]" in cmd, cmd


def test_probability_maps_default_to_linear(anat_and_moving, tmp_path):
    anat, moving = anat_and_moving
    affine = tmp_path / "a.mat"
    affine.write_bytes(b"")

    with mock.patch.object(subprocess, "run") as run:
        propagate_anat_image(anat, moving, affine, tmp_path / "o.nii.gz")

    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--interpolation") + 1] == "Linear"


def test_labels_can_request_nearest_neighbour(anat_and_moving, tmp_path):
    anat, moving = anat_and_moving
    affine = tmp_path / "a.mat"
    affine.write_bytes(b"")

    with mock.patch.object(subprocess, "run") as run:
        propagate_anat_image(anat, moving, affine, tmp_path / "o.nii.gz",
                             interpolation="NearestNeighbor")

    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--interpolation") + 1] == "NearestNeighbor"


def test_nonlinear_warp_is_prepended(anat_and_moving, tmp_path):
    """ANTs applies right-to-left, so the inverse warp must precede the affine."""
    anat, moving = anat_and_moving
    affine = tmp_path / "a.mat"
    affine.write_bytes(b"")
    warp = tmp_path / "1InverseWarp.nii.gz"
    warp.write_bytes(b"")

    with mock.patch.object(subprocess, "run") as run:
        propagate_anat_image(anat, moving, affine, tmp_path / "o.nii.gz",
                             inverse_warp=warp)

    cmd = run.call_args[0][0]
    assert cmd.index(str(warp)) < cmd.index(f"[{affine},1]")


def test_absent_inverse_warp_is_ignored(anat_and_moving, tmp_path):
    anat, moving = anat_and_moving
    affine = tmp_path / "a.mat"
    affine.write_bytes(b"")

    with mock.patch.object(subprocess, "run") as run:
        propagate_anat_image(anat, moving, affine, tmp_path / "o.nii.gz",
                             inverse_warp=tmp_path / "missing.nii.gz")

    cmd = run.call_args[0][0]
    assert "missing.nii.gz" not in " ".join(cmd)


def test_resample_from_to_would_miss_the_anatomy(anat_and_moving):
    """Why the registration is needed: document what the old path did.

    Not a test of our code -- a guard on the premise. If these frames ever did
    agree, the registration would be redundant and this test would fail loudly.
    """
    from nibabel.processing import resample_from_to
    anat, moving = anat_and_moving

    anat_img = nib.load(anat)
    moving_img = nib.load(moving)
    naive = resample_from_to(anat_img, moving_img, order=1)

    src = np.asarray(anat_img.dataobj) > 0.5
    got = np.asarray(naive.dataobj) > 0.5
    overlap = (src & got).sum() / max(src.sum(), 1)
    assert overlap < 0.5, (
        f"frames agree (overlap {overlap:.1%}); the offset premise no longer holds"
    )


def test_propagate_anat_mask_returns_the_transform_for_reuse():
    """aCompCor depends on getting the affine back, so pin the contract."""
    import inspect
    from neurofaune.preprocess.utils import registration_utils

    src = inspect.getsource(registration_utils.propagate_anat_mask)
    assert "'moving_to_anat_affine'" in src
    assert "'inverse_warp'" in src
