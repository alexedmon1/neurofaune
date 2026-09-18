"""Order of the -t arguments when a registration is applied in reverse.

The inverse of an ANTs registration (0GenericAffine + 1Warp) is
`-t [0GenericAffine.mat,1] -t 1InverseWarp`, affine FIRST. Several propagation
functions -- and a test that pinned them -- had it the other way round, reading
"ANTs applies transforms right-to-left" as "put the warp first". Checked against
the InverseWarped image antsRegistration writes itself, the affine-first order
reproduces it exactly (r=1.000) and warp-first does not (r=0.966 on a
subject->template registration, r=0.666 on template->SIGMA). Every propagated
parcellation went through the wrong order.
"""
import subprocess
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.registration_utils import propagate_anat_mask
from neurofaune.templates.anat_registration import (
    propagate_atlas_direct,
    propagate_atlas_to_anat,
)
from neurofaune.templates.sigma_warp import inverse_transform_args


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _ants_writes_output(cmd, *args, **kwargs):
    """Stand-in for subprocess.run: write a small labelled image to -o."""
    if cmd[0] == 'antsApplyTransforms':
        data = np.zeros((6, 6, 4), np.int16)
        data[2:4, 2:4, 1:3] = 1
        nib.save(nib.Nifti1Image(data, np.eye(4)), cmd[cmd.index('-o') + 1])
    return subprocess.CompletedProcess(cmd, 0, '', '')


def _transforms(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == '-t']


def test_helper_puts_the_inverted_affine_first(tmp_path):
    affine, warp = _touch(tmp_path / 'x_0GenericAffine.mat'), _touch(tmp_path / 'x_1InverseWarp.nii.gz')
    assert inverse_transform_args(affine, warp) == [f'[{affine},1]', str(warp)]


def test_helper_drops_an_absent_or_missing_warp(tmp_path):
    affine = tmp_path / 'a.mat'
    assert inverse_transform_args(affine) == [f'[{affine},1]']
    assert inverse_transform_args(affine, tmp_path / 'missing.nii.gz') == [f'[{affine},1]']


def test_atlas_to_anat_chain_order(tmp_path):
    sub, ses = 'sub-01', 'ses-1'
    xf = tmp_path / 'transforms' / sub / ses
    s_aff = _touch(xf / f'{sub}_{ses}_T2w_to_template_0GenericAffine.mat')
    s_inv = _touch(xf / f'{sub}_{ses}_T2w_to_template_1InverseWarp.nii.gz')
    tpl = tmp_path / 'templates' / 'anat' / '1' / 'transforms'
    t_aff = _touch(tpl / 'tpl-to-SIGMA_0GenericAffine.mat')
    t_inv = _touch(tpl / 'tpl-to-SIGMA_1InverseWarp.nii.gz')
    ref = tmp_path / 'ref.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((6, 6, 4), np.float32), np.eye(4)), ref)

    with mock.patch.object(subprocess, 'run', side_effect=_ants_writes_output) as run:
        propagate_atlas_to_anat(tmp_path / 'atlas.nii.gz', ref, tmp_path / 'transforms',
                                tmp_path / 'templates', sub, ses,
                                tmp_path / 'out.nii.gz', generate_qc=False)
    assert _transforms(run.call_args[0][0]) == [
        f'[{s_aff},1]', str(s_inv),        # subject <- template
        f'[{t_aff},1]', str(t_inv),        # template <- SIGMA
    ]


def test_atlas_direct_chain_order(tmp_path):
    sub, ses = 'sub-01', 'ses-1'
    xf = tmp_path / sub / ses
    aff = _touch(xf / f'{sub}_{ses}_T2w_to_SIGMA_0GenericAffine.mat')
    inv = _touch(xf / f'{sub}_{ses}_T2w_to_SIGMA_1InverseWarp.nii.gz')
    ref = tmp_path / 'ref.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((6, 6, 4), np.float32), np.eye(4)), ref)

    with mock.patch.object(subprocess, 'run', side_effect=_ants_writes_output) as run:
        propagate_atlas_direct(tmp_path / 'atlas.nii.gz', ref, tmp_path, sub, ses,
                               tmp_path / 'out.nii.gz', generate_qc=False)
    assert _transforms(run.call_args[0][0]) == [f'[{aff},1]', str(inv)]


@pytest.mark.parametrize('nonlinear', [False, True])
def test_anat_mask_to_moving_chain_order(tmp_path, nonlinear):
    """The DWI second mask uses nonlinear=True, so it went through the bad order."""
    work = tmp_path / 'work'
    prefix = work / 'mask_moving_to_anat_'
    aff = _touch(Path(f'{prefix}0GenericAffine.mat'))
    inv = _touch(Path(f'{prefix}1InverseWarp.nii.gz'))

    with mock.patch.object(subprocess, 'run') as run:
        propagate_anat_mask(moving_ref=tmp_path / 'b0.nii.gz',
                            anat_t2w=tmp_path / 't2w.nii.gz',
                            anat_mask=tmp_path / 'mask.nii.gz',
                            out_mask=tmp_path / 'out.nii.gz', work_dir=work,
                            nonlinear=nonlinear)
    apply_cmd = run.call_args_list[-1][0][0]
    expected = [f'[{aff},1]'] + ([str(inv)] if nonlinear else [])
    assert _transforms(apply_cmd) == expected
