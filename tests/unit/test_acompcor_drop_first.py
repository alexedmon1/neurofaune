"""aCompCor: discarding the leading (global-signal) PC of each tissue.

The leading PC of a large tissue mask is the global signal rather than
tissue-specific physiology. Measured on the cuprizone rat cohort,
corr(PC1, brain-mean) was 0.980 (WM) and 0.984 (CSF) while PC2 onward sat near
zero -- so with global signal regression also enabled the same nuisance was
modelled three times, and the CSF and WM regressor sets looked interchangeable
(max |r| 0.997 across all 52 sessions) even though the informative components
were not.
"""
import numpy as np
import nibabel as nib

from neurofaune.preprocess.utils.func.acompcor import extract_acompcor_components

SHAPE = (18, 18, 18)
N_TP = 60
AFFINE = np.eye(4)


def _write(path, data):
    nib.save(nib.Nifti1Image(data.astype(np.float32), AFFINE), str(path))
    return path


def _mask(tmp_path, name, lo, hi):
    d = np.zeros(SHAPE)
    d[lo:hi, lo:hi, lo:hi] = 1.0
    return _write(tmp_path / f"{name}.nii.gz", d)


def _bold_with_global_signal(tmp_path):
    """Every voxel carries a strong shared drift plus its own noise.

    That shared term is what PC1 of any sizeable mask locks onto -- the very
    thing global signal regression already removes.
    """
    rng = np.random.default_rng(0)
    gs = 20.0 * np.sin(np.linspace(0, 6 * np.pi, N_TP))
    data = rng.normal(100.0, 1.0, size=SHAPE + (N_TP,)) + gs
    return _write(tmp_path / "bold.nii.gz", data)


def _env(tmp_path):
    bold = _bold_with_global_signal(tmp_path)
    csf = _mask(tmp_path, "csf", 2, 8)
    wm = _mask(tmp_path, "wm", 10, 16)
    brain = _write(tmp_path / "brain.nii.gz", np.ones(SHAPE))
    return bold, csf, wm, brain


def test_regressor_count_is_preserved(tmp_path):
    """Dropping PC1 must not quietly return one fewer regressor."""
    bold, csf, wm, brain = _env(tmp_path)
    kw = dict(bold_file=bold, csf_mask=csf, wm_mask=wm, n_components=4,
              erode_voxels=0, brain_mask=brain)
    keep = extract_acompcor_components(**kw)
    drop = extract_acompcor_components(**kw, drop_first_component=True)
    assert keep['components'].shape[1] == drop['components'].shape[1] == 8
    assert drop['n_components_csf'] == drop['n_components_wm'] == 4


def test_dropping_removes_the_global_component(tmp_path):
    """What survives must be less global than what was there before."""
    bold, csf, wm, brain = _env(tmp_path)
    kw = dict(bold_file=bold, csf_mask=csf, wm_mask=wm, n_components=3,
              erode_voxels=0, brain_mask=brain)
    gs = nib.load(str(bold)).get_fdata().reshape(-1, N_TP).mean(axis=0)

    def max_gs_corr(res):
        c = res['components']
        return max(abs(np.corrcoef(c[:, j], gs)[0, 1]) for j in range(c.shape[1]))

    assert max_gs_corr(extract_acompcor_components(**kw)) > 0.9
    assert max_gs_corr(
        extract_acompcor_components(**kw, drop_first_component=True)) < 0.5


def test_dropped_component_globalness_is_reported(tmp_path):
    """Don't assume PC1 is global -- record how global it was, per session."""
    bold, csf, wm, brain = _env(tmp_path)
    res = extract_acompcor_components(
        bold_file=bold, csf_mask=csf, wm_mask=wm, n_components=3,
        erode_voxels=0, brain_mask=brain, drop_first_component=True)
    corr = res['dropped_component_gs_corr']
    assert set(corr) == {'CSF', 'WM'}
    assert all(v > 0.9 for v in corr.values()), corr
    assert res['drop_first_component'] is True


def test_columns_carry_the_true_pc_index(tmp_path):
    """The first retained regressor is PC2; calling it comp_1 erases that."""
    bold, csf, wm, brain = _env(tmp_path)
    out = tmp_path / "acompcor.tsv"
    extract_acompcor_components(
        bold_file=bold, csf_mask=csf, wm_mask=wm, n_components=3,
        erode_voxels=0, brain_mask=brain, drop_first_component=True,
        output_file=out)
    header = out.read_text().splitlines()[0].split('\t')
    assert header[:3] == ['csf_comp_2', 'csf_comp_3', 'csf_comp_4']
    assert 'csf_comp_1' not in header and 'wm_comp_1' not in header


def test_default_is_unchanged_behaviour(tmp_path):
    """Existing studies must not silently change their nuisance model."""
    bold, csf, wm, brain = _env(tmp_path)
    res = extract_acompcor_components(
        bold_file=bold, csf_mask=csf, wm_mask=wm, n_components=3,
        erode_voxels=0, brain_mask=brain, output_file=tmp_path / "a.tsv")
    assert res['drop_first_component'] is False
    assert (tmp_path / "a.tsv").read_text().splitlines()[0].startswith('csf_comp_1')
