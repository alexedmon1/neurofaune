#!/usr/bin/env python3
"""
Unit tests for the brain-mask refinement pipeline stage.

The image logic is tested in test_atlas_guided_strip.py. What is pinned here is the
wiring: the config key selects the stage, the atlas->subject chain is in the order
that round-trips, and the default never touches desc-brain_mask.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.workflows import anat_mask_refinement as amr


def _config(study_root, **refine):
    atlas = study_root / 'atlas'
    return {
        'paths': {'study_root': str(study_root)},
        'atlas': {'study_space': {
            'brain_mask': str(atlas / 'mask.nii.gz'),
            'template_masked': str(atlas / 'template.nii.gz'),
            'parcellation': str(atlas / 'labels.nii.gz')}},
        'anatomical': {'skull_strip': {'refine': {'method': 'atlas_iterative', **refine}}},
    }


def _save(data, path, zooms=(1.25, 1.25, 8.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data), np.diag([*zooms, 1.0]))
    img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return path


def _study(tmp_path):
    """A session with a detached bright 'bulb' the first-pass mask missed."""
    shape = (32, 32, 12)
    sub, ses = 'sub-01', 'ses-1'
    raw = np.full(shape, 10.0, np.float32)
    parcellation = np.zeros(shape, np.int16)
    raw[8:22, 8:22, 3:9] = 100.0
    parcellation[8:22, 8:22, 3:9] = 1
    raw[22:26, 12:17, 4:8] = 95.0
    parcellation[22:26, 12:17, 4:8] = 2
    initial = np.zeros(shape, np.uint8)
    initial[6:22, 6:24, 3:9] = 1              # over-inclusive, and misses the bulb

    anat = tmp_path / 'derivatives' / sub / ses / 'anat'
    bids = tmp_path / 'raw' / 'bids' / sub / ses / 'anat'
    _save(raw, bids / f'{sub}_{ses}_run-1_T2w.nii.gz')
    _save(raw[:, :, :5], bids / f'{sub}_{ses}_run-2_T2w.nii.gz')   # scout, wrong grid
    _save(initial, anat / f'{sub}_{ses}_desc-brain_mask.nii.gz')
    _save(raw * initial * 1000, anat / f'{sub}_{ses}_desc-preproc_T2w.nii.gz')
    _save(raw * initial, anat / f'{sub}_{ses}_desc-skullstrip_T2w.nii.gz')
    _save(parcellation, anat / f'{sub}_{ses}_atlas-SIGMA_dseg.nii.gz')
    _save(raw, tmp_path / 'work' / sub / ses / 'anat' / f'{sub}_{ses}_T2w_n4.nii.gz')
    _save((parcellation == 1).astype(np.uint8), tmp_path / 'atlas' / 'mask.nii.gz')

    xf = tmp_path / 'transforms' / sub / ses
    xf.mkdir(parents=True)
    (xf / f'{sub}_{ses}_T2w_to_template_0GenericAffine.mat').touch()
    (xf / f'{sub}_{ses}_T2w_to_template_1InverseWarp.nii.gz').touch()
    tpl = tmp_path / 'templates' / 'anat' / '1' / 'transforms'
    tpl.mkdir(parents=True)
    (tpl / 'tpl-to-SIGMA_0GenericAffine.mat').touch()
    (tpl / 'tpl-to-SIGMA_1InverseWarp.nii.gz').touch()
    return sub, ses, anat, sorted(bids.glob('*_T2w.nii.gz'))


@pytest.fixture
def fake_ants(monkeypatch):
    """Stand in for antsApplyTransforms: the 'warp' is the identity."""
    calls = []

    def apply_chain(source, reference, chain, output, interpolation='NearestNeighbor'):
        calls.append(list(chain))
        src = nib.load(str(source))
        nib.save(nib.Nifti1Image(np.asarray(src.dataobj), src.affine), str(output))
        return Path(output)

    monkeypatch.setattr(amr, 'apply_chain', apply_chain)
    return calls


def test_settings_default_to_disabled_without_config():
    assert amr.get_refine_settings({})['method'] == 'none'


def test_settings_read_config_and_merge_qc():
    s = amr.get_refine_settings({'anatomical': {'skull_strip': {'refine': {
        'method': 'atlas_iterative', 'iterations': 3, 'apply': True,
        'qc': {'volume_mm3': [1000, 3000]}}}}})
    assert s['iterations'] == 3 and s['apply'] is True
    assert s['qc']['volume_mm3'] == (1000, 3000)
    assert s['qc']['tissue_fraction'] == amr.DEFAULT_QC['tissue_fraction']


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match='refine.method'):
        amr.get_refine_settings({'anatomical': {'skull_strip': {'refine': {'method': 'bet'}}}})


def test_chain_inverts_each_leg_affine_first(tmp_path):
    """The order that round-trips: [aff,1], inv-warp per leg, subject leg first."""
    sub, ses, _, _ = _study(tmp_path)
    chain = amr.atlas_to_subject_chain(tmp_path / 'transforms', tmp_path / 'templates',
                                       sub, ses)
    assert chain[0].startswith('[') and 'T2w_to_template_0GenericAffine' in chain[0]
    assert 'T2w_to_template_1InverseWarp' in chain[1]
    assert chain[2].startswith('[') and 'tpl-to-SIGMA_0GenericAffine' in chain[2]
    assert 'tpl-to-SIGMA_1InverseWarp' in chain[3]


def test_chain_missing_transform_returns_none(tmp_path):
    assert amr.atlas_to_subject_chain(tmp_path, tmp_path, 'sub-x', 'ses-1') is None


def test_disabled_method_does_nothing(tmp_path):
    sub, ses, anat, raw = _study(tmp_path)
    cfg = _config(tmp_path, method='none')
    row = amr.run_brain_mask_refinement(cfg, sub, ses, tmp_path, raw)
    assert row['status'] == 'disabled'
    assert not (anat / f'{sub}_{ses}_desc-refinedbrain_mask.nii.gz').exists()


def test_default_writes_alongside_and_leaves_original(tmp_path, fake_ants):
    sub, ses, anat, raw = _study(tmp_path)
    original = np.asarray(nib.load(str(anat / f'{sub}_{ses}_desc-brain_mask.nii.gz')).dataobj)
    row = amr.run_brain_mask_refinement(_config(tmp_path), sub, ses, tmp_path, raw,
                                        montage=False)
    assert row['status'] == 'refined' and row['applied'] is False
    assert row['raw_t2w'].endswith('run-1_T2w.nii.gz')      # the scout is skipped
    assert row['n_registrations'] == 0                        # iterations=1
    refined = nib.load(str(anat / f'{sub}_{ses}_desc-refinedbrain_mask.nii.gz')).get_fdata() > 0
    assert refined[23, 14, 6]                                 # bulb recovered
    assert not refined[6, 6, 5]                               # over-inclusion trimmed
    after = np.asarray(nib.load(str(anat / f'{sub}_{ses}_desc-brain_mask.nii.gz')).dataobj)
    np.testing.assert_array_equal(after, original)
    assert not (anat / f'{sub}_{ses}_desc-initialbrain_mask.nii.gz').exists()


def test_apply_replaces_mask_keeps_initial_and_restrips(tmp_path, fake_ants):
    sub, ses, anat, raw = _study(tmp_path)
    mask_p = anat / f'{sub}_{ses}_desc-brain_mask.nii.gz'
    original = np.asarray(nib.load(str(mask_p)).dataobj).copy()
    cfg = _config(tmp_path, apply=True)
    row = amr.run_brain_mask_refinement(cfg, sub, ses, tmp_path, raw, montage=False)
    assert row['status'] == 'refined' and row['applied'] is True

    initial_p = anat / f'{sub}_{ses}_desc-initialbrain_mask.nii.gz'
    np.testing.assert_array_equal(np.asarray(nib.load(str(initial_p)).dataobj), original)
    mask = nib.load(str(mask_p)).get_fdata() > 0
    preproc = nib.load(str(anat / f'{sub}_{ses}_desc-preproc_T2w.nii.gz')).get_fdata()
    assert preproc[23, 14, 6] == pytest.approx(95.0 * 1000)   # recovered tissue present
    assert np.all(preproc[~mask] == 0)

    # A second applied run must still compare against the true first-pass mask.
    again = amr.run_brain_mask_refinement(cfg, sub, ses, tmp_path, raw, montage=False)
    assert again['volume_before_mm3'] == pytest.approx(row['volume_before_mm3'])
    np.testing.assert_array_equal(np.asarray(nib.load(str(initial_p)).dataobj), original)


def test_missing_parcellation_is_skipped_not_raised(tmp_path, fake_ants):
    sub, ses, anat, raw = _study(tmp_path)
    (anat / f'{sub}_{ses}_atlas-SIGMA_dseg.nii.gz').unlink()
    row = amr.run_brain_mask_refinement(_config(tmp_path), sub, ses, tmp_path, raw)
    assert row['status'] == 'skipped' and 'parcellation' in row['reason']


def test_summary_counts_only_refined_rows(tmp_path, fake_ants):
    sub, ses, _, raw = _study(tmp_path)
    row = amr.run_brain_mask_refinement(_config(tmp_path), sub, ses, tmp_path, raw,
                                        montage=False)
    table = amr.write_refinement_summary(
        [row, {'subject': 'sub-02', 'session': 'ses-1', 'status': 'skipped'}],
        tmp_path / 'qc')
    import json
    summary = json.loads((tmp_path / 'qc' / 'mask_refinement_summary.json').read_text())
    assert table.exists() and summary['n_sessions'] == 1 and summary['n_skipped'] == 1
    assert amr.write_refinement_summary([], tmp_path / 'none') is None
